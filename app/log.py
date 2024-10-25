import logging
from logging.config import dictConfig
from pydantic import BaseModel
from app.config import settings
import os

# Create log directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

class LogConfig(BaseModel):
    LOGGER_NAME: str = "app"
    LOG_APP_FORMAT: str = "%(levelname)s | APP | %(asctime)s | %(message)s"
    # LOG_API_FORMAT: str = "%(levelname)s | API | %(asctime)s | %(request_line)s - %(status_code)s"
    LOG_API_FORMAT: str = "%(levelname)s | API | %(asctime)s | %(message)s"
    LOG_LEVEL: str = settings.LOG_LEVEL.upper()
    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "format": LOG_APP_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "format": LOG_API_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "default_colored": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_APP_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
        "access_colored": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": LOG_API_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
        "access": {
            "formatter": "access",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
        "console_default": {
            "formatter": "default_colored",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "console_access": {
            "formatter": "access_colored",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    }
    loggers: dict = {
        LOGGER_NAME: {"handlers": ["default", "console_default"], "level": LOG_LEVEL, "propagate": False},
        "uvicorn": {"handlers": ["default", "console_default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access", "console_access"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default", "console_default"], "level": "INFO", "propagate": False},
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

dictConfig(LogConfig().model_dump())
logger = logging.getLogger("app")
