from typing import Annotated
from typing_extensions import TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from qdrant_client import QdrantClient

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

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that provides summary answer based on a context in the form of a collection of spotify reviews.
            each row represents a single review
            If user's question does not relate to spotify's review, respond with 'this question is irrelevant to spotify reviews analysis'
            If the collection of reviews is not relevant or doesn't contain the information needed to answer the question, respond with 'NO_ANSWER'.
            If the collection of reviews is relevant, provide a concise answer based on it."""),
            ("human", "Reviews:\n{context}\n\nQuestion: {question}")
        ])

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

    def _find_contextual_response(self, query: str):
        embedding = get_embedding(query)

        response = self._query_qdrant(self.collection, embedding)
        if response:
            llm_response = self._get_llm_response(response, query)
            if llm_response != "NO_ANSWER":
                return llm_response

        return "i cant formulate an answer based on the context provided"

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
        print(f"TOTAL REVIEWS: {len(reviews)}")
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

    def run_chat(self):
        print("Chatbot initialized. Type 'quit' to exit.")
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            self.stream_graph_updates(user_input)

# Instantiate and run ReviewChatbot
review_chatbot = ReviewChatbot()
review_chatbot.run_chat()
