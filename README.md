
---

# Spotify Review Chatbot

This repository provides a Spotify review chatbot, built to analyze and answer user inquiries based on Spotify's Google Play Store reviews.

**Dataset:**  
The full dataset for this project is available on Kaggle: [3.4 Million Spotify Google Store Reviews](https://www.kaggle.com/datasets/bwandowando/3-4-million-spotify-google-store-reviews).

## Requirements

- **Docker**
- **Docker Compose**
- **Poetry** (Python 3.12.3)

---

## Getting Started

### Install Dependencies

To install project dependencies, run:
```bash
poetry install
```

---

## Knowledge Base

The knowledge base is powered by Qdrant as the vector database, which stores the semantic embeddings of review texts.

- **Vector Database:**  
  [Qdrant](https://qdrant.tech/documentation/quickstart) serves as the vector database for this chatbot. To deploy Qdrant, run:
  ```bash
  docker-compose -f docker-compose-qdrant.yml up
  ```

- **Embedding Model:**  
  We use [TinyBERT](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2) for semantic representation of review texts. TinyBERT is served with Huggingface’s Optimum ONNX and transformed into vector representations using Transformers' `feature-extraction` pipeline.

  To deploy TinyBERT:
  1. **Build Optimum ONNX Serving Image:**
     ```bash
     cd tinybert/serving
     docker build -t optimum-onnx-serving-cpu:0.1.2 .
     ```
  2. **Update `model.onnx` Path in Docker Compose Volumes**  
     Example:
     ```yaml
     - /home/miftah/Downloads/job_application/mekari/review_bot/tinybert:/app/models
     ```
  3. **Start the TinyBERT Docker Compose:**
     ```bash
     docker-compose up
     ```

Once Qdrant and TinyBERT are deployed successfully, load the dataset:
1. Download the dataset: [SPOTIFY_REVIEWS.csv](https://drive.usercontent.google.com/download?id=1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM&export=download&authuser=0&confirm=t&uuid=6e16677f-518a-4234-a40b-fa2fcf5c7f72&at=AN_67v0zAA_AXLxQ-CUszJFdfeOp%3A1729829750160).
2. Update `fname` in `qdrant_scripts/load_data.py` to point to the dataset path.
3. Run the data loading script:
   ```bash
   poetry run python load_data.py
   ```

---

## Chatbot

The chatbot uses [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/) as the main library, integrating [LangChain](https://python.langchain.com/docs/introduction/) to manage AzureOpenAI and `ChatPromptTemplate` for seamless dialog flow.

### Run Chatbot Interactively
To run the chatbot in terminal mode:
```bash
./scripts/run_rag.sh
```

### Deploy Chatbot API
To deploy the chatbot API:
```bash
./scripts/run_api.sh
```

### Deploy Streamlit UI
To deploy the UI (depends on the API):
```bash
./scripts/run_ui.sh
```

---

## API Endpoints

The following endpoints are available in this chatbot:

1. **POST `/ask`**  
   **Description:** Responds to user inquiries related to Spotify reviews.  
   **Input:** JSON object with `user_input` (text).  
   **Example Input:**
   ```json
   {
     "user_input": "In comparison to our application, which music streaming platform are users most likely to compare ours with?"
   }
   ```
   **Example Output:**
   ```json
   {
     "response": "Users are most likely to compare our application with Pandora."
   }
   ```

2. **PUT `/review/<review_id>`**  
   **Description:** Inserts a review item into the Qdrant collection.  
   **Input:** JSON dictionary with review details.  
   **Example Input:**
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
   **Example Output:**
   ```json
   {
     "status": "item id 19206 is inserted into spotify_review"
   }
   ```

3. **DELETE `/review/<item_id>`**  
   **Description:** Deletes a review item from the Qdrant collection based on the item ID.  
   **Example Output:**
   ```json
   {
     "status": "item id 19206 is deleted from spotify_review"
   }
   ```

4. **GET `/collections`**  
   **Description:** Lists all available collections in Qdrant.  
   **Example Output:**
   ```json
   {
     "result": [
       {
         "collection_name": "spotify_review",
         "documents_count": 8443
       }
     ]
   }
   ```

5. **GET `/healthcheck`**  
   **Description:** Verifies if the API is running.  
   **Example Output:**
   ```json
   {
     "status": "ok"
   }
   ```

To test these endpoints, open [Swagger UI](http://localhost:<API_PORT>/docs) in your browser.

--- 


