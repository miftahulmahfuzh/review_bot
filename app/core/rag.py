from typing_extensions import TypedDict
from typing import Annotated
import re

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from ollama import Client
import tiktoken

from app.core.utils import get_embedding
from app.config import settings
from app.log import logger

class State(TypedDict):
    messages: Annotated[list, add_messages]
    scores: Annotated[list[float], "Scores for responses"]

class ReviewChatbot:
    def __init__(self):
        self.client = Client(host=f"{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}")
        self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection = settings.REVIEW_COLLECTION_NAME
        self.graph = self._build_graph()

        self.system_desc = open(settings.SYSTEM_DESCRIPTION).read()
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_desc),
            ("human", "Reviews:\n{context}\n\nQuestion: {question}")
        ])
        self.scoring_system_desc = open(settings.SCORING_SYSTEM_DESCRIPTION).read()
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", self.scoring_system_desc),
            ("human", "Reviews:\n{context}\n\nQuestion: {question}\nAnswer: {response}")
        ])
        self._last_context = ""

        self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # Keep for length estimation
        self.max_tokens = 8192
        self.max_response_tokens = 500
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B", trust_remote_code=True)
        self.max_tokens = 8192
        self.max_response_tokens = 500

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

    def _shorten_context(self, context: str, query: str) -> str:
        """
        Shorten the response context to fit within token limits while preserving the most relevant content.
        Uses AutoTokenizer to accurately count tokens and maintain context within limits.
        """
        # Calculate token counts for fixed components
        system_tokens = len(self.tokenizer.encode(self.system_desc))
        query_tokens = len(self.tokenizer.encode(query))

        # Calculate available tokens for reviews
        available_tokens = self.max_tokens - system_tokens - query_tokens - self.max_response_tokens

        # Split reviews and process them while respecting token limits
        reviews = context.split('\n')
        shortened_reviews = []
        current_tokens = 0

        # Process each review
        for review in reviews:
            # Skip empty reviews
            if not review.strip():
                continue

            # Count tokens for current review
            review_tokens = len(self.tokenizer.encode(review))

            # Check if adding this review would exceed the token limit
            if current_tokens + review_tokens <= available_tokens:
                shortened_reviews.append(review)
                current_tokens += review_tokens
            else:
                # If we can't add any more reviews, break the loop
                break

        # Log the reduction in review count
        logger.info(f"Shortened reviews from {len(reviews)} to {len(shortened_reviews)}")

        # If no reviews could be included, return a warning message
        if not shortened_reviews:
            logger.warning("Could not include any reviews within token limit")
            return "The context is too long to process. Please try a more specific question."

        return '\n'.join(shortened_reviews)

    def _find_contextual_response(self, query: str):
        embedding = get_embedding(query)
        context = self._query_qdrant(self.collection, embedding)
        self._last_context = context
        if context:
            try:
                llm_response = self._get_llm_response(context, query)
                return llm_response
            except Exception as e:
                if "context length exceeded" in str(e).lower():
                    logger.warning(f"Token length exceeded, attempting to shorten context.")
                    shortened_context = self._shorten_response(context, query)
                    self._last_context = shortened_context
                    try:
                        llm_response = self._get_llm_response(shortened_context, query)
                        return llm_response
                    except Exception as inner_e:
                        logger.error(f"Error after shortening context: {str(inner_e)}")
                        return "I encountered an error processing your query. Please try rephrasing or simplifying your question."
                else:
                    logger.error(f"Ollama Error: {str(e)}")
                    return "I encountered an unexpected error. Please try again."

        return "I can't formulate an answer based on the context provided."

    def _convert_literal_percentage_to_float(self, value: str) -> float:
        # Remove any non-numeric or non-symbol characters except '/', '%', and '.'
        value = re.sub(r"[^0-9/%\.]", "", value)

        if '/' in value:  # handle '10/10' or '4.8/5' format
            try:
                numerator, denominator = map(float, value.split('/'))
                return round(numerator / denominator, 2)
            except (ValueError, ZeroDivisionError):
                raise ValueError("Invalid fraction format or division by zero.")
        elif '%' in value:  # handle '85%' format
            try:
                result = float(value.replace('%', '')) / 100
                return result
            except ValueError:
                raise ValueError("Invalid percentage format.")
        else:
            raise ValueError("Unsupported format. Expected '10/10' or '85%'.")

    def _normalize_score(self, current_score, context):
        score = current_score
        if score == 100:
            score = 1
        elif score > 100:
            total_reviews = len(context.split("\n"))
            score /= total_reviews
        elif 1 < score < 10:
            score /= 10
        return score

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

            ollama_response = self.client.chat(
                model=settings.OLLAMA_CHAT_MODEL,
                options={"temperature": settings.TEMPERATURE},
                messages=[
                    {
                        "role": "system",
                        "content": self.scoring_system_desc
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nCalculate score for the above response. Don't give any explanation"
                    }
                ]
            )

            try:
                c = ollama_response['message']['content'].strip()
                score = float(c)
                score = self._normalize_score(score, context)
                result = round(score, 2)
                return result
            except ValueError as e:
                try:
                    c = c.split()[-1]
                    score = float(c)
                    score = self._normalize_score(score, context)
                except ValueError as e:
                    try:
                        score = self._convert_literal_percentage_to_float(c.split()[-1])
                    except ValueError as e:
                        logger.error(f"Error converting score to float: {str(e)}")

        except Exception as e:
            logger.error(f"Error calculating response score: {str(e)}")

        return score

    def _get_llm_response(self, context: str, query: str):
        prompt = self.qa_prompt.format(
            context=context,
            question=query
        )

        try:
            ollama_response = self.client.chat(
                model=settings.OLLAMA_CHAT_MODEL,
                options={"temperature": settings.TEMPERATURE},
                messages=[
                    {
                        "role": "system",
                        "content": self.system_desc
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return ollama_response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error during Ollama response generation: {str(e)}")
            return "An error occurred while generating the response."

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
