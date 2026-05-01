"""
Real-time emotion prediction from facial images or video frames.

Provides high-level interface for macro and micro-expression detection.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ..models import EmotionNet, EmotionNetConfig


class EmotionPredictor:
    """
    High-level emotion prediction interface.
    
    Supports:
    - Single image inference
    - Video stream processing
    - Batch inference
    - Macro and micro-expression detection
    """
    
    MACRO_EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    
    MICRO_EMOTIONS = [
        "positive",
        "negative",
        "surprise",
        "repression",
        "others",
    ]
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        config: Optional[EmotionNetConfig] = None,
    ):
        """
        Initialize emotion predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            device: "cuda" or "cpu" (auto-detected if None)
            config: Model configuration (uses default if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        if config is None:
            config = EmotionNetConfig()
        
        self.model = EmotionNet(config).to(self.device)
        self.config = config
        
        # Load weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Model file not found at {model_path}. Using random weights.")
        
        self.model.eval()
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            target_size: Target image size
        
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict_macro_emotion(
        self,
        image: np.ndarray,
    ) -> Dict[str, any]:
        """
        Predict macro-expression emotion.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with predictions:
                - emotion: Predicted emotion name
                - confidence: Confidence score
                - scores: Confidence scores for all classes
                - class_names: Names of all emotion classes
        """
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model.forward(tensor)
        
        predictions = self.model.get_macro_predictions(tensor)
        pred_idx = predictions["labels"].item()
        
        return {
            "emotion": self.MACRO_EMOTIONS[pred_idx],
            "confidence": predictions["confidence"].item(),
            "scores": predictions["predictions"][0].cpu().numpy(),
            "class_names": self.MACRO_EMOTIONS,
            "emotion_idx": pred_idx,
        }
    
    def predict_micro_emotion(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, any]:
        """
        Predict micro-expression emotion.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence to consider it a micro-expression
        
        Returns:
            Dictionary with predictions:
                - emotion: Predicted micro-emotion name
                - classification_confidence: Classification confidence
                - detection_confidence: Detection confidence (involuntary leak score)
                - is_micro_expression: Boolean whether detected as genuine micro-expression
                - scores: Confidence scores for all classes
                - class_names: Names of all micro-emotion classes
        """
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model.forward(tensor)
        
        predictions = self.model.get_micro_predictions(tensor)
        pred_idx = predictions["labels"].item()
        
        detection_confidence = predictions["detection_confidence"].item()
        is_micro_expr = detection_confidence >= confidence_threshold
        
        return {
            "emotion": self.MICRO_EMOTIONS[pred_idx],
            "classification_confidence": predictions["classification_confidence"].item(),
            "detection_confidence": detection_confidence,
            "is_micro_expression": is_micro_expr,
            "scores": predictions["predictions"][0].cpu().numpy(),
            "class_names": self.MICRO_EMOTIONS,
            "emotion_idx": pred_idx,
        }
    
    def predict_both(
        self,
        image: np.ndarray,
        micro_confidence_threshold: float = 0.5,
    ) -> Dict[str, any]:
        """
        Predict both macro and micro-expressions.
        
        Args:
            image: Input image
            micro_confidence_threshold: Minimum confidence for micro-expression
        
        Returns:
            Dictionary with both predictions
        """
        macro_pred = self.predict_macro_emotion(image)
        micro_pred = self.predict_micro_emotion(image, micro_confidence_threshold)
        
        return {
            "macro": macro_pred,
            "micro": micro_pred,
        }
    
    def process_video_frame(
        self,
        frame: np.ndarray,
        detect_type: str = "both",
    ) -> Dict[str, any]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame
            detect_type: "macro", "micro", or "both"
        
        Returns:
            Prediction results
        """
        if detect_type == "macro":
            return self.predict_macro_emotion(frame)
        elif detect_type == "micro":
            return self.predict_micro_emotion(frame)
        else:
            return self.predict_both(frame)
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get model configuration and size info."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_class": self.model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": self.config.__dict__,
            "device": self.device,
        }
