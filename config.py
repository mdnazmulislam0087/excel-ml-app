import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
    STORAGE_DIR = BASE_DIR / "storage"
    UPLOAD_DIR = STORAGE_DIR / "uploads"
    MODEL_DIR = STORAGE_DIR / "models"
    PLOT_DIR = BASE_DIR / "webapp" / "static" / "plots"

    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB upload limit

    # Training settings
    MLP_MAX_EPOCHS = 300
    MLP_HIDDEN = (64, 64)
    RANDOM_STATE = 42
