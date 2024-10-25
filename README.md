---

# Spotify Review Chatbot

This repository contains a Spotify Review Chatbot that processes user inquiries and provides insights based on 3.4 million Spotify reviews from the Google Play Store. The full dataset can be accessed [here](https://www.kaggle.com/datasets/bwandowando/3-4-million-spotify-google-store-reviews).

## Requirements

- **Docker**
- **Docker-Compose**
- **Poetry** (Python 3.12.3)

---

## Setup

### Dependencies
To install the necessary dependencies, run:
```bash
poetry install
```

### Knowledge Base

The chatbot leverages Qdrant as a vector database and TinyBERT for semantic understanding of review text.

#### 1. Qdrant Setup
Qdrant serves as the vector storage for embeddings, which allows for efficient similarity searches. To deploy Qdrant, execute:
```bash
docker-compose -f docker-compose-qdrant.yml up
```

#### 2. TinyBERT Setup
TinyBERT, a distilled model of BERT, provides a semantic representation of review text, using [Huggingface's TinyBERT model](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2). The model is served via Optimum ONNX, which offers fast inference on CPU.

To deploy TinyBERT:
1. Build the Docker image for the ONNX server:
    ```bash
    cd tinybert/serving
    docker build -t optimum-onnx-serving-cpu:0.1.2 .
    ```

2. Update the path to the `model.onnx` file in the `docker-compose.yml`:
    ```yaml
    volumes:
      - /path/to/your/model:/app/models
    ```

3. Start the ONNX server:
    ```bash
    docker-compose up
    ```

#### 3. Data Loading
Once both Qdrant and TinyBERT are running, load the dataset:
1. Download the [SPOTIFY_REVIEWS.csv](https://drive.usercontent.google.com/download?id=1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM&export=download&authuser=0&confirm=t&uuid=6e16677f-518a-4234-a40b-fa2fcf5c7f72&at=AN_67v0zAA_AXLxQ-CUszJFdfeOp%3A1729829750160).
2. Update the `fname` variable in `qdrant_scripts/load_data.py` to the path of the CSV.
3. Load data into Qdrant:
    ```bash
    poetry run python qdrant_scripts/load_data.py
    ```

---

## Chatbot

The chatbot is implemented using [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/) as the main library, with [LangChain](https://python.langchain.com/docs/introduction/) for AzureOpenAI and ChatPromptTemplate integration.

### Running the Chatbot

To interact with the chatbot through the terminal, use:
```bash
./scripts/run_rag.sh
```

For API deployment, run:
```bash
./scripts/run_api.sh
```

To launch the Streamlit UI (requires the chatbot API to be running), execute:
```bash
./scripts/run_ui.sh
```

---

## API Endpoints

The chatbot includes five main endpoints:

1. **POST /ask**
   Answers user inquiries related to Spotify reviews.
   **Input**
   ```json
   { "user_input": "In comparison to our application, which music streaming platform are users most likely to compare ours with?" }
   ```
   **Output**
   ```json
   { "response": "Users are most likely to compare our application with Pandora." }
   ```

2. **PUT /review/<review_id>**
   Inserts a new review item into the Qdrant collection.
   **Input**
   ```json
   {
     "_id": 0,
     "review_id": "14a011a8-7544-47b4-8480-c502af0ac26f",
     "pseudo_author_id": "152618553977019693742",
     "author_name": "A Google user",
     "review_text": "Use it every day",
     "review_rating": 5,
     "review_likes": 1,
     "author_app_version": "1.1.0.91",
     "review_timestamp": "2014-05-27 14:21:48"
   }
   ```
   **Output**
   ```json
   { "status": "item id 19206 is inserted into spotify_review" }
   ```

3. **DELETE /review/<item_id>**
   Deletes a review item from the Qdrant collection based on the `item_id`.
   **Output**
   ```json
   { "status": "item id 19206 is deleted from spotify_review" }
   ```

4. **GET /collections**
   Lists all available collections in Qdrant.
   **Output**
   ```json
   { "result": [ { "collection_name": "spotify_review", "documents_count": 8443 } ] }
   ```

5. **GET /healthcheck**
   Health check endpoint to verify the status of the API.
   **Output**
   ```json
   { "status": "ok" }
   ```

---

This completes the setup and usage guide for the Spotify Review Chatbot. Enjoy exploring insights from millions of Spotify user reviews!
