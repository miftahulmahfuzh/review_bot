from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import SecretStr
import os

class Config(BaseSettings):

    LOG_LEVEL: str = "INFO"

    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    TOPN: int = 100
    REVIEW_COLLECTION_NAME: str = "spotify_review"

    EMBEDDING_SERVING_URL: str = "http://10.181.131.244:8899/forward"
    CACHE_QUERY_EMBEDDING: bool = True
    VECTOR_SIZE: int = 1024

    API_KEY: SecretStr = "ebce2698dadf0593c979a2798c84e49a0"
    API_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8092

    AZURE_ENDPOINT: SecretStr
    AZURE_API_KEY: SecretStr
    AZURE_API_VERSION: SecretStr
    AZURE_DEPLOYMENT_NAME: SecretStr
    TEMPERATURE: float

    SYSTEM_DESCRIPTION: Path = "config/system_description.txt"
    SCORING_SYSTEM_DESCRIPTION: Path = "config/scoring_system_description.txt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Config()  # type: ignore

__import__("pprint").pprint(settings.__dict__)

if settings.CACHE_QUERY_EMBEDDING:
    try:
        os.mkdir(".cache")
    except FileExistsError:
        pass
