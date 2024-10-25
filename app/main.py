import uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from app import schema
from app.config import settings
# from app.core.review import Review
from app.core.document_db import db

app = FastAPI(
    title="Review Chatbot API",
    version=settings.API_VERSION,
    description="API for spotify review chatbot"
)

# review = Review()

# Dependency to check x-api-key
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY.get_secret_value():  # Compare with the API key from settings
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.put("/review/{item_id}",
         response_model=schema.InsertOutput,
         dependencies=[Depends(verify_api_key)]
         )
def insert_review(item_id: int, payload: schema.Review):
    result = db.insert(item_id, payload, settings.REVIEW_COLLECTION_NAME)
    return schema.InsertOutput(**result)

@app.delete("/review/{item_id}",
            response_model=schema.DeleteOutput,
            dependencies=[Depends(verify_api_key)]
            )
def delete_article(item_id: int):
    result = climate.delete_article(item_id)
    return schema.DeleteOutput(**result)

@app.get("/collections", response_model=schema.Knowledges)
def get_collections():
    result = db.knowledges()
    return schema.Knowledges(**result)

@app.get("/healthcheck", summary="API health check", description="This endpoint returns the current status of the API.")
def healthcheck():
    return JSONResponse(status_code=200, content={"status": "ok"})

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        port=settings.API_PORT,
        host=settings.API_HOST
    )

