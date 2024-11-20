from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Doodle Recognition"

    # Model settings
    MODEL_PATH: str = "models/doodle_model.pth"
    IMAGE_SIZE: int = 28
    NUM_CLASSES: int = 15

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173"]

    # Data settings
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # Ensure directories exist
    def create_directories(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAW_DATA_DIR.mkdir(exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

    class Config:
        case_sensitive = True
        env_file = ".env"


# Initialize settings
settings = Settings()
settings.create_directories()
