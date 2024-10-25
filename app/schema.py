from pydantic import BaseModel, Field
from typing import List

class Review(BaseModel):
    _id: int
    review_id: str
    pseudo_author_id: str
    author_name: str
    review_text: str
    review_rating: int
    review_likes: int
    author_app_version: str
    review_timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class Knowledges(BaseModel):
    class Knowledge(BaseModel):
        collection_name: str
        documents_count: int

    result: List[Knowledge]

class InsertOutput(BaseModel):
    status: str = Field(..., description="status explanation for insert item", example="item id 19206 is inserted into spotify_review")

class DeleteOutput(BaseModel):
    status: str = Field(..., description="status explanation for deleted item", example="item id 19206 is deleted from spotify_review")
