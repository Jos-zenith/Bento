"""API endpoints for emotion recognition."""

from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from affective_intelligence.inference import EmotionPredictor


# Initialize router
router = APIRouter(prefix="/api/v1/emotion", tags=["emotion"])

# Global predictor (loaded on startup)
predictor: Optional[EmotionPredictor] = None


class EmotionPredictionResponse(BaseModel):
    """Response model for emotion prediction."""
    
    emotion: str
    confidence: float
    class_scores: dict
    type: str  # "macro" or "micro"


class DualEmotionPredictionResponse(BaseModel):
    """Response model for dual emotion prediction."""
    
    macro_emotion: str
    macro_confidence: float
    micro_emotion: str
    micro_confidence: float
    is_micro_expression: bool
    micro_detection_confidence: float


def init_emotion_predictor(model_path: str):
    """Initialize emotion predictor on app startup."""
    global predictor
    predictor = EmotionPredictor(model_path=model_path)


@router.post("/predict/macro", response_model=EmotionPredictionResponse)
async def predict_macro_emotion(file: UploadFile = File(...)):
    """
    Predict macro-expression emotion from image.
    
    - **file**: Image file (JPEG, PNG)
    
    Returns emotion prediction with confidence score.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Emotion predictor not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Predict
        result = predictor.predict_macro_emotion(image)
        
        return EmotionPredictionResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            class_scores={name: score for name, score in zip(
                result["class_names"],
                result["scores"]
            )},
            type="macro",
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/micro", response_model=EmotionPredictionResponse)
async def predict_micro_emotion(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
):
    """
    Predict micro-expression emotion from image.
    
    - **file**: Image file (JPEG, PNG)
    - **confidence_threshold**: Minimum confidence for genuine micro-expression
    
    Returns micro-emotion prediction with detection confidence.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Emotion predictor not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Predict
        result = predictor.predict_micro_emotion(image, confidence_threshold)
        
        return EmotionPredictionResponse(
            emotion=result["emotion"],
            confidence=result["classification_confidence"],
            class_scores={name: score for name, score in zip(
                result["class_names"],
                result["scores"]
            )},
            type="micro",
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/dual", response_model=DualEmotionPredictionResponse)
async def predict_dual_emotion(file: UploadFile = File(...)):
    """
    Predict both macro and micro-expression emotions.
    
    - **file**: Image file (JPEG, PNG)
    
    Returns both macro and micro-emotion predictions with confidence scores.
    Useful for detecting involuntary emotional leaks.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Emotion predictor not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Predict
        result = predictor.predict_both(image)
        
        return DualEmotionPredictionResponse(
            macro_emotion=result["macro"]["emotion"],
            macro_confidence=result["macro"]["confidence"],
            micro_emotion=result["micro"]["emotion"],
            micro_confidence=result["micro"]["classification_confidence"],
            is_micro_expression=result["micro"]["is_micro_expression"],
            micro_detection_confidence=result["micro"]["detection_confidence"],
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """Get emotion recognition model information."""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Emotion predictor not initialized")
    
    return predictor.get_model_summary()
