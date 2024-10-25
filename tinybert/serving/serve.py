import ast
import json
import logging
import os
import re
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union

import torch
import uvicorn
import numpy as np
from dotenv import dotenv_values
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTModelForFeatureExtraction,
)
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline

values = {**dotenv_values("../config/.env"), **os.environ}

ONNX_MODEL_DIR = "./models"
ONNX_MODEL_NAME = values["ONNX_MODEL_NAME"]
MAX_SEQ_LENGTH = int(values["MAX_SEQ_LENGTH"])

app = FastAPI()

class Request(BaseModel):
    pass

class SingleDocument(Request):
    inputs: List[str]

    class Config:
        schema_extra = {
            "example": {
                "inputs": ["SENTENCE A", "SENTENCE B", "SENTENCE C"],
            }
        }

class PairDocument(Request):
    inputs: List[Dict]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [{ "text": "SENTENCE A", "text_pair":"SENTENCE B"}],
            }
        }

class ResponseItem(BaseModel):
    output: Union[List[List[str]], List[List[float]]]

    class Config:
        schema_extra = {
            "example": {
                "output": [
                    ["neutral"], ["positive"], ["negative"]
                ],
            }
        }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc) -> JSONResponse:
    data = {
        "message": "Input data is invalid. text->List[str]"
    }
    return JSONResponse(
        content=data, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

class Classifier:

    def __init__(self, model_path):
        with open(model_path + "/config.json") as json_file:
            config = json.load(json_file)
        self.task = config["finetuning_task"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=MAX_SEQ_LENGTH)

        if self.task == 'token_cls':
            model = ORTModelForTokenClassification.from_pretrained(model_path, file_name=ONNX_MODEL_NAME)
        elif self.task == 'embedding':
            model = ORTModelForFeatureExtraction.from_pretrained(model_path, file_name=ONNX_MODEL_NAME)
        else:
            model = ORTModelForSequenceClassification.from_pretrained(model_path, file_name=ONNX_MODEL_NAME)

        if self.task == 'token_cls':
            self.onnx_classifier = pipeline("token-classification", model=model, tokenizer=self.tokenizer, truncation=True)
        elif self.task == 'doc_multi_cls':
            self.onnx_classifier = pipeline("text-classification", model=model, tokenizer=self.tokenizer, return_all_scores=True, truncation=True)
        elif self.task == 'embedding':
            self.onnx_classifier = model
        else:
            self.onnx_classifier = pipeline("text-classification", model=model, tokenizer=self.tokenizer, truncation=True)

    def __mask_score_in_token(self, token: Dict) -> Dict:
        if "score" in token:
            token["score"] = float(token["score"])
        return token

    def __mask_score_in_instance(self, instance: List[Dict]) -> List[Dict]:
        instance = map(self.__mask_score_in_token, instance)
        return list(instance)

    def get_word_index(self, text: str) -> List:
        res = [(ele.start(), ele.end()) for ele in re.finditer(r'\S+', text)]
        return res

    def __to_index2entity(self, item: Dict) -> Dict:
        res = { item["start"]: item["entity"] }
        return res

    def remove_non_alphanumeric(
        self, word: str, start: int
    ) -> Tuple[str, Tuple[int, int]]:
        tmp = re.sub(r'[\W_]+', ' ', word)
        filtered_word = tmp.strip()
        end = start + len(filtered_word)
        start_end = (start, end)
        return filtered_word, start_end

    def __get_label_from_instance(
        self,
        text_instance: List[Tuple]) -> List[Dict]:
        text = text_instance[0]
        instance = text_instance[1]
        word_index = self.get_word_index(text)

        tmp = list(map(self.__to_index2entity, instance))
        model_index2entity = {k:v for d in tmp for k, v in d.items()}

        res = []
        for start_end in word_index:
            start = start_end[0]
            end = start_end[1]
            if start in model_index2entity:
                word = text[start:end]
                word, start_end = self.remove_non_alphanumeric(word, start)
                predicted_label = model_index2entity[start]
                res.append({
                    "word": word,
                    "index": start_end,
                    "label": predicted_label
                })
        return res

    def get_embeddings(self, list_sentences: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(list_sentences, padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = self.onnx_classifier(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings.tolist()

    def predict(self, list_sentences: List):
        def format_output(predicted_class):
            if self.task in ['doc_cls', 'doc_pair_cls']:
                output = predicted_class
            elif self.task == 'token_cls':
                output = list(map(self.__mask_score_in_instance, predicted_class))
                list_text_instance = zip(list_sentences, predicted_class)
                aligned_output = list(map(self.__get_label_from_instance, list_text_instance))
                output = aligned_output
            elif self.task == 'embedding':
                output = self.get_embeddings(list_sentences)
            else:
                thresh = 0.5
                output = []
                for pclass in predicted_class:
                    pred = {dict_label["label"]: dict_label["score"] for dict_label in pclass if dict_label["score"] > thresh}
                    output.append(pred)
            return output

        if self.task == 'embedding':
            return format_output(list_sentences)
        else:
            predicted_class = self.onnx_classifier(list_sentences)
            return format_output(predicted_class)

clf = Classifier(ONNX_MODEL_DIR)

@app.post("/forward", response_model=ResponseItem)
async def get_inputs(api_input: Union[SingleDocument, PairDocument], status_code=status.HTTP_200_OK):
    benchmark_start = timer()
    list_input = api_input.inputs
    serving_predictions = clf.predict(list_input)
    serving_result = {"output": serving_predictions}
    benchmark_finish = timer() - benchmark_start
    return JSONResponse(content=serving_result)

if __name__ == '__main__':
    uvicorn.run("onnx_optimizer.serve:app", port=3110, host="0.0.0.0", reload=True)
