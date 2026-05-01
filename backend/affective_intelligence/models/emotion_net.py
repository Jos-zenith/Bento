"""
EfficientNet-B0 based facial emotion recognition model.

Supports both macro-expressions (FER2013) and micro-expressions (CASME II, SAMM).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


@dataclass
class EmotionNetConfig:
    """Configuration for EmotionNet model."""
    
    num_macro_classes: int = 7  # FER2013: neutral, happy, sad, angry, fearful, disgusted, surprised
    num_micro_classes: int = 5  # CASME II/SAMM: positive, negative, surprise, repression, others
    pretrained: bool = True
    dropout_rate: float = 0.3
    use_attention: bool = True
    embedding_dim: int = 512


class EmotionNet(nn.Module):
    """
    EfficientNet-B0 based emotion recognition network.
    
    Detects both macro-expressions (spontaneous, intentional) and micro-expressions
    (involuntary emotional leaks). Uses separate classification heads for each type.
    """
    
    def __init__(self, config: EmotionNetConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained EfficientNet-B0
        backbone = efficientnet_b0(pretrained=config.pretrained)
        
        # Extract features from backbone (all but classification head)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Get the number of features from backbone
        num_backbone_features = 1280  # EfficientNet-B0 output channels
        
        # Shared embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(num_backbone_features, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
        )
        
        # Attention module for micro-expression detection
        if config.use_attention:
            self.attention = SpatialAttention(num_backbone_features)
        
        # Macro-expression head (e.g., FER2013 - 7 classes)
        self.macro_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_macro_classes),
        )
        
        # Micro-expression head (e.g., CASME II/SAMM - 5 classes)
        self.micro_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_micro_classes),
        )
        
        # Confidence head for micro-expression detection reliability
        self.confidence_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_embeddings: Whether to return embedding vectors
        
        Returns:
            Dictionary with:
                - macro_logits: (B, num_macro_classes)
                - micro_logits: (B, num_micro_classes)
                - confidence: (B, 1) - confidence for micro-expression detection
                - embeddings: (B, embedding_dim) if return_embeddings=True
        """
        # Backbone features
        features = self.features(x)  # (B, 1280, H', W')
        
        # Apply attention if enabled
        if self.config.use_attention:
            features = self.attention(features) * features
        
        # Global average pooling
        pooled = self.avgpool(features)  # (B, 1280, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 1280)
        
        # Shared embedding
        embeddings = self.embedding(pooled)  # (B, embedding_dim)
        
        # Classification heads
        macro_logits = self.macro_head(embeddings)  # (B, num_macro_classes)
        micro_logits = self.micro_head(embeddings)  # (B, num_micro_classes)
        confidence = self.confidence_head(embeddings)  # (B, 1)
        
        output = {
            "macro_logits": macro_logits,
            "micro_logits": micro_logits,
            "confidence": confidence,
        }
        
        if return_embeddings:
            output["embeddings"] = embeddings
        
        return output
    
    def get_macro_predictions(self, x: torch.Tensor) -> dict:
        """Get macro-expression predictions only."""
        output = self.forward(x)
        macro_preds = torch.softmax(output["macro_logits"], dim=1)
        return {
            "predictions": macro_preds,
            "labels": torch.argmax(macro_preds, dim=1),
            "confidence": torch.max(macro_preds, dim=1)[0],
        }
    
    def get_micro_predictions(self, x: torch.Tensor) -> dict:
        """Get micro-expression predictions with confidence scores."""
        output = self.forward(x)
        micro_preds = torch.softmax(output["micro_logits"], dim=1)
        confidence = output["confidence"].squeeze(1)
        
        return {
            "predictions": micro_preds,
            "labels": torch.argmax(micro_preds, dim=1),
            "classification_confidence": torch.max(micro_preds, dim=1)[0],
            "detection_confidence": confidence,
        }
    
    def get_both_predictions(self, x: torch.Tensor) -> dict:
        """Get both macro and micro-expression predictions."""
        output = self.forward(x)
        
        macro_preds = torch.softmax(output["macro_logits"], dim=1)
        micro_preds = torch.softmax(output["micro_logits"], dim=1)
        confidence = output["confidence"].squeeze(1)
        
        return {
            "macro": {
                "predictions": macro_preds,
                "labels": torch.argmax(macro_preds, dim=1),
                "confidence": torch.max(macro_preds, dim=1)[0],
            },
            "micro": {
                "predictions": micro_preds,
                "labels": torch.argmax(micro_preds, dim=1),
                "classification_confidence": torch.max(micro_preds, dim=1)[0],
                "detection_confidence": confidence,
            },
        }


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on relevant facial regions."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        channel_out = (avg_out + max_out).view(b, c, 1, 1)
        return self.sigmoid(channel_out)
