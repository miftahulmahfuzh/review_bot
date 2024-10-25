from typing import Tuple
from app.log import logger
from requests import post
import json
import hashlib
import os
import pickle

from app.config import settings

def predict_doc_cls(url:str, text:str) -> Tuple[str, float]:
    data = {
        "inputs": [text]
    }
    r = None
    label = "unknown"
    confidence = 0
    try:
        r = post(url, data=json.dumps(data)).json()
        r = r["output"][0]
        label = r["label"]
        confidence = r["score"]
    except Exception as e:
        logger.error(f"FAILED DURING API CALL (predict_doc_cls). url: {url}")
        logger.error(e)
    return (label, confidence)

def predict_doc_pair_cls(url:str, text_a:str, text_b:str) -> Tuple[str, float]:
    data = {
        "inputs": [{"text":text_a, "text_pair":text_b}]
    }
    r = None
    label = "unknown"
    confidence = 0
    try:
        r = post(url, data=json.dumps(data)).json()
        r = r["output"][0]
        label = r["label"]
        confidence = r["score"]
    except Exception as e:
        logger.error(f"FAILED DURING API CALL (predict_doc_pair_cls). url: {url}")
        logger.error(e)
    return (label, confidence)

def predict_doc_multi_cls(url:str, text:str) -> dict:
    data = {
        "inputs": [text]
    }
    r = {}
    try:
        r = post(url, data=json.dumps(data)).json()
        r = r["output"][0]
    except Exception as e:
        logger.error(f"FAILED DURING API CALL (predict_doc_multi_cls). url: {url}")
        logger.error(e)
    return r

def vector(text):
    headers = {"content-type": "application/json"}
    data = {"inputs": [text]}
    r = post(settings.EMBEDDING_SERVING_URL, data=json.dumps(data), headers=headers)
    r = r.json()["output"][0]
    return r

def get_embedding(text: str):
    if settings.CACHE_QUERY_EMBEDDING:
        cache_key = hashlib.md5(text.strip().encode()).hexdigest()
        if cache_key in os.listdir(".cache"):
            with open(f".cache/{cache_key}", "rb") as f:
                # logger.info("use cached embedding")
                return pickle.load(f)

    # logger.info("creating embedding..")
    embedding = vector(text)

    if settings.CACHE_QUERY_EMBEDDING:
        cache_key = hashlib.md5(text.strip().encode()).hexdigest()
        with open(f".cache/{cache_key}", "wb") as f:
            pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding
