# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import torch
from pathlib import Path
from app.model.cnn import DoodleNet
from app.utils.preprocessing import DoodlePreprocessor

# Initialize FastAPI app
app = FastAPI(title="Doodle Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels (15 categories)
CLASS_LABELS = [
    "apple", "banana", "cat", "dog", "elephant",
    "fish", "guitar", "house", "lion", "pencil",
    "pizza", "rabbit", "snake", "spider", "tree"
]

# Initialize model and paths
MODEL_PATH = Path("models/doodle_model.pth")
model = DoodleNet(num_classes=len(CLASS_LABELS))

# Load model if exists
if MODEL_PATH.exists():
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

# Initialize preprocessor
preprocessor = DoodlePreprocessor()


class DrawingInput(BaseModel):
    points: List[List[float]]


class PredictionOutput(BaseModel):
    predictions: List[Tuple[str, float]]
    success: bool
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Doodle Recognition API is running",
        "model_loaded": MODEL_PATH.exists(),
        "num_classes": len(CLASS_LABELS),
        "available_categories": CLASS_LABELS
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "num_classes": len(CLASS_LABELS),
        "available_categories": CLASS_LABELS,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/api/v1/predict", response_model=PredictionOutput)
async def predict_drawing(drawing: DrawingInput):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Process drawing
        input_tensor = preprocessor.process_drawing(drawing.points)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]

            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, k=3)

            # Format predictions
            predictions = [
                (CLASS_LABELS[idx], prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]

        return PredictionOutput(
            predictions=predictions,
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
