from requests import post
import json
import hashlib
import os
import pickle

from app.config import settings
from app.log import logger
from ollama import Client

def vector_ollama(text: str):
    try:
        ollama_local = Client(host="http://0.0.0.0:11435")
        r = ollama_local.embed(
            settings.OLLAMA_EMBEDDING_MODEL,
            text
        )
        result = r["embeddings"][0]
        return result
    except RequestException as e:
        logger.error(f"Failed to connect to embedding service: {str(e)}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Failed to process embedding response: {str(e)}")
        return None

def vector(text: str):
    try:
        headers = {"content-type": "application/json"}
        data = {"inputs": [text]}
        r = post(settings.EMBEDDING_SERVING_URL, data=json.dumps(data), headers=headers)
        r.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
        return r.json()["output"][0]
    except RequestException as e:
        logger.error(f"Failed to connect to embedding service: {str(e)}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Failed to process embedding response: {str(e)}")
        return None

def get_embedding(text: str):
    if settings.CACHE_QUERY_EMBEDDING:
        cache_key = hashlib.md5(text.strip().encode()).hexdigest()
        if cache_key in os.listdir(settings.CACHE_DIR):
            with open(f"{settings.CACHE_DIR}/{cache_key}", "rb") as f:
                # logger.info("use cached embedding")
                return pickle.load(f)

    # logger.info("creating embedding..")
    if settings.USE_OLLAMA_EMBEDDING:
        embedding = vector_ollama(text)
    else:
        embedding = vector(text)

    if settings.CACHE_QUERY_EMBEDDING:
        cache_key = hashlib.md5(text.strip().encode()).hexdigest()
        with open(f"{settings.CACHE_DIR}/{cache_key}", "wb") as f:
            pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding
