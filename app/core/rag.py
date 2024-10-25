from typing import Annotated
from typing_extensions import TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from qdrant_client import QdrantClient
import  openai
import tiktoken

from app.log import logger
from app.config import settings
from app.core.utils import get_embedding

class State(TypedDict):
    messages: Annotated[list, add_messages]

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
        # Initialize tokenizer for GPT-4
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 8192  # GPT-4's maximum context length
        self.max_response_tokens = 500  # Reserve tokens for the response

    def _build_graph(self):
        def chatbot(state: State):
            query = state["messages"][-1].content
            response = self._find_contextual_response(query)
            return {"messages": [response]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile()

    def _shorten_response(self, response: str, query: str) -> str:
        """
        Shorten the response context to fit within token limits while preserving the most relevant content.
        """
        # Calculate tokens for the system description and query
        system_tokens = len(self.tokenizer.encode(self.system_desc))
        query_tokens = len(self.tokenizer.encode(query))

        # Calculate available tokens for the response context
        available_tokens = self.max_tokens - system_tokens - query_tokens - self.max_response_tokens

        # Split response into reviews
        reviews = response.split('\n')

        # Initialize shortened response
        shortened_reviews = []
        current_tokens = 0

        # Add reviews until we approach the token limit
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
        if response:
            try:
                llm_response = self._get_llm_response(response, query)
                return llm_response
            except openai.BadRequestError as e:
                if "maximum context length" in str(e).lower():
                    logger.warning(f"Token length exceeded, attempting to shorten response: {str(e)}")
                    shortened_response = self._shorten_response(response, query)
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

        return "I can't formulate an answer based on the context provided."

    def _query_qdrant(self, collection_name: str, embedding):
        results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=settings.TOPN
        )
        text = ""
        iid = 0
        reviews = []
        for result in results:
            r = result.payload
            text = r['review_text']
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

        self.graph = self._build_graph()  # Re-build the graph to reset memory
        return response

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
