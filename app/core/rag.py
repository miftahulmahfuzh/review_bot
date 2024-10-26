from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
import tiktoken
import openai

from app.core.utils import get_embedding
from app.config import settings
from app.log import logger

class State(TypedDict):
    messages: Annotated[list, add_messages]
    scores: Annotated[list[float], "Scores for responses"]

class ReviewChatbot:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_ENDPOINT.get_secret_value(),
            openai_api_key=settings.AZURE_API_KEY.get_secret_value(),
            openai_api_version=settings.AZURE_API_VERSION.get_secret_value(),
            azure_deployment=settings.AZURE_DEPLOYMENT_NAME.get_secret_value(),
            temperature=settings.TEMPERATURE,
        )
        self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection = settings.REVIEW_COLLECTION_NAME
        self.graph = self._build_graph()

        self.system_desc = open(settings.SYSTEM_DESCRIPTION).read()
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_desc),
            ("human", "Reviews:\n{context}\n\nQuestion: {question}")
        ])
        scoring_system_desc = open(settings.SCORING_SYSTEM_DESCRIPTION).read()
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", scoring_system_desc),
            ("human", "Reviews:\n{context}\n\nQuestion: {question}\nAnswer: {response}")
        ])
        self._last_context = ""

        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 8192  # GPT-4's maximum context length
        self.max_response_tokens = 500  # Reserve tokens for the response

    def _build_graph(self):
        def chatbot(state: State):
            query = state["messages"][-1].content
            response = self._find_contextual_response(query)
            is_irrelevant = "this question is irrelevant" in response.lower()
            score = 1
            if not is_irrelevant:
                score = self._calculate_response_score(response, query, self._last_context)
            logger.info(f"Response score: {score}")
            return {"messages": [response], "scores": [score]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile()

    def _calculate_response_score(self, response: str, query: str, context: str) -> float:
        """
        Calculate a score for the response based on how well it represents the reviews.
        """
        score = 0
        try:
            prompt = self.scoring_prompt.format(
                context=context,
                question=query,
                response=response
            )

            llm_response = self.llm.invoke([("system", prompt), ("user", f"Calculate score for the above response. Dont give any explanation")])

            try:
                c = llm_response.content.strip()
                score = float(c)
                return round(score, 2)
            except ValueError as e:
                try:
                    c = c.split()[-1]
                    score = float(c)
                except ValueError as e:
                    logger.error(f"Error converting score to float: {str(e)}")

        except openai.APITimeoutError as e:
            logger.error(f"OpenAI API timeout error in scoring: {str(e)}")

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error in scoring: {str(e)}")

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error in scoring: {str(e)}")

        except Exception as e:
            logger.error(f"Error calculating response score: {str(e)}")

        return score

    def _shorten_response(self, response: str, query: str) -> str:
        """
        Shorten the response context to fit within token limits while preserving the most relevant content.
        """
        system_tokens = len(self.tokenizer.encode(self.system_desc))
        query_tokens = len(self.tokenizer.encode(query))

        available_tokens = self.max_tokens - system_tokens - query_tokens - self.max_response_tokens

        reviews = response.split('\n')

        shortened_reviews = []
        current_tokens = 0

        for review in reviews:
            review_tokens = len(self.tokenizer.encode(review))
            if current_tokens + review_tokens <= available_tokens:
                shortened_reviews.append(review)
                current_tokens += review_tokens
            else:
                break

        logger.info(f"Shortened reviews from {len(reviews)} to {len(shortened_reviews)}")
        return '\n'.join(shortened_reviews)

    def _find_contextual_response(self, query: str):
        embedding = get_embedding(query)
        response = self._query_qdrant(self.collection, embedding)
        self._last_context = response
        if response:
            try:
                llm_response = self._get_llm_response(response, query)
                return llm_response
            except openai.BadRequestError as e:
                if "maximum context length" in str(e).lower():
                    logger.warning(f"Token length exceeded, attempting to shorten response: {str(e)}")
                    shortened_response = self._shorten_response(response, query)
                    self._last_context = shortened_response
                    try:
                        llm_response = self._get_llm_response(shortened_response, query)
                        return llm_response
                    except Exception as inner_e:
                        logger.error(f"Error after shortening response: {str(inner_e)}")
                        return "I encountered an error processing your query. Please try rephrasing or simplifying your question."
                else:
                    logger.error(f"Unexpected OpenAI error: {str(e)}")
                    return "I encountered an unexpected error. Please try again."
            except Exception as e:
                logger.error(f"General error in finding contextual response: {str(e)}")
                return "I encountered an error processing your query. Please try again."

            except openai.APITimeoutError as e:
                logger.error(f"OpenAI API timeout error: {str(e)}")
                return "The request timed out. Please try again in a moment."

            except openai.APIConnectionError as e:
                logger.error(f"OpenAI API connection error: {str(e)}")
                return "I'm having trouble connecting to the service. Please check your internet connection and try again."

            except openai.AuthenticationError as e:
                logger.error(f"OpenAI authentication error: {str(e)}")
                return "There's an issue with the service authentication. Please contact support."

            except Exception as e:
                logger.error(f"Unexpected error in LLM processing: {str(e)}")
                return "An unexpected error occurred. Please try again."

        return "I can't formulate an answer based on the context provided."

    def _query_qdrant(self, collection_name: str, embedding):
        results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=settings.TOPN
        )
        reviews = []
        for result in results:
            r = result.payload
            text = r['review_text']
            text = " ".join(text.split())
            reviews.append(text)
        logger.info(f"TOTAL REVIEWS: {len(reviews)}")
        text = "\n".join(reviews)
        return text

    def _get_llm_response(self, context: str, query: str):
        # Prepare the chat prompt using the ChatPromptTemplate
        prompt = self.qa_prompt.format(
            context=context,
            question=query
        )

        # Get the response from the LLM
        llm_response = self.llm.invoke([("system", prompt), ("user", query)])
        return llm_response.content.strip()

    def stream_graph_updates(self, user_input: str):
        for event in self.graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1])

    def ask(self, user_input: str):
        response = None
        for event in self.graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                response = value["messages"][-1]
                score = value["scores"][-1]

        self.graph = self._build_graph()  # Re-build the graph to reset memory
        result = {"response": response, "score": score}
        return result

    def run_chat(self):
        print("Chatbot initialized. Type 'quit' to exit.")
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            self.stream_graph_updates(user_input)

if __name__=="__main__":
    review_chatbot = ReviewChatbot()
    review_chatbot.run_chat()
