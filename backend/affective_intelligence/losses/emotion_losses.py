"""
Custom loss functions for emotion recognition.

Includes:
- Focal Loss for handling class imbalance
- Center Loss for micro-expression embedding learning
- Combined loss for multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Particularly useful for micro-expressions which are rare/hard to detect.
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy/hard examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Predicted logits of shape (B, C)
            targets: Target class indices of shape (B,)
        
        Returns:
            Focal loss value
        """
        # Get softmax probabilities
        p = F.softmax(predictions, dim=1)
        
        # Get class probabilities
        class_mask = F.one_hot(targets, num_classes=predictions.size(1))
        p_t = (p * class_mask).sum(1)
        
        # Compute focal term
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class CenterLoss(nn.Module):
    """
    Center Loss for learning better embeddings.
    
    Helps micro-expression embeddings cluster around class centers,
    improving discrimination between similar micro-expressions.
    Reference: Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """
    
    def __init__(self, num_classes: int, embedding_dim: int, alpha: float = 0.5):
        """
        Initialize Center Loss.
        
        Args:
            num_classes: Number of emotion classes
            embedding_dim: Dimensionality of embeddings
            alpha: Learning rate for updating centers
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # Initialize class centers
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_uniform_(self.centers, a=1)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            embeddings: Batch of embeddings of shape (B, embedding_dim)
            targets: Target class indices of shape (B,)
        
        Returns:
            Center loss value
        """
        # Get target centers
        target_centers = self.centers[targets]  # (B, embedding_dim)
        
        # Compute distance from embeddings to target centers
        loss = torch.norm(embeddings - target_centers, dim=1) ** 2
        
        return loss.mean()
    
    def update_centers(self, embeddings: torch.Tensor, targets: torch.Tensor):
        """Update class centers (can be called during training)."""
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = targets == i
                if mask.sum() > 0:
                    center_delta = self.alpha * (self.centers[i] - embeddings[mask].mean(0))
                    self.centers[i] -= center_delta


class CombinedEmotionLoss(nn.Module):
    """
    Combined loss for multi-task emotion recognition.
    
    Combines:
    - Cross-entropy for macro-expression classification
    - Focal loss for micro-expression classification (handles rarity)
    - Center loss for better embedding learning
    - Confidence loss for micro-expression detection reliability
    """
    
    def __init__(
        self,
        num_macro_classes: int,
        num_micro_classes: int,
        embedding_dim: int,
        use_focal: bool = True,
        use_center: bool = True,
        use_confidence: bool = True,
        lambda_macro: float = 1.0,
        lambda_micro: float = 1.5,
        lambda_center: float = 0.5,
        lambda_confidence: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        center_alpha: float = 0.5,
    ):
        """
        Initialize Combined Loss.
        
        Args:
            num_macro_classes: Number of macro-expression classes
            num_micro_classes: Number of micro-expression classes
            embedding_dim: Dimensionality of embeddings
            use_focal: Whether to use focal loss for micro-expressions
            use_center: Whether to use center loss
            use_confidence: Whether to use confidence loss
            lambda_macro: Weight for macro-expression loss
            lambda_micro: Weight for micro-expression loss
            lambda_center: Weight for center loss
            lambda_confidence: Weight for confidence loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            center_alpha: Alpha parameter for center loss
        """
        super().__init__()
        
        self.lambda_macro = lambda_macro
        self.lambda_micro = lambda_micro
        self.lambda_center = lambda_center
        self.lambda_confidence = lambda_confidence
        
        # Macro-expression: standard cross-entropy
        self.macro_loss = nn.CrossEntropyLoss()
        
        # Micro-expression: focal loss (handles rarity) or cross-entropy
        if use_focal:
            self.micro_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.micro_loss = nn.CrossEntropyLoss()
        
        # Center loss for better embeddings
        if use_center:
            self.center_loss = CenterLoss(
                num_classes=num_micro_classes,
                embedding_dim=embedding_dim,
                alpha=center_alpha,
            )
        else:
            self.center_loss = None
        
        # Confidence loss (BCE for reliability)
        if use_confidence:
            self.confidence_loss = nn.BCELoss()
        else:
            self.confidence_loss = None
    
    def forward(
        self,
        macro_logits: torch.Tensor,
        micro_logits: torch.Tensor,
        macro_targets: torch.Tensor,
        micro_targets: torch.Tensor,
        confidence: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            macro_logits: Macro-expression predictions (B, num_macro_classes)
            micro_logits: Micro-expression predictions (B, num_micro_classes)
            macro_targets: Macro-expression targets (B,)
            micro_targets: Micro-expression targets (B,)
            confidence: Confidence scores (B, 1)
            embeddings: Embedding vectors (B, embedding_dim)
        
        Returns:
            Dictionary with loss values
        """
        # Macro-expression loss
        macro_loss = self.macro_loss(macro_logits, macro_targets)
        
        # Micro-expression loss
        micro_loss = self.micro_loss(micro_logits, micro_targets)
        
        # Center loss
        center_loss = 0.0
        if self.center_loss is not None:
            center_loss = self.center_loss(embeddings, micro_targets)
        
        # Confidence loss (assume high confidence for detected micro-expressions)
        confidence_loss = 0.0
        if self.confidence_loss is not None:
            # Use micro-targets to determine if micro-expression is present
            confidence_targets = (micro_targets > 0).float().unsqueeze(1)
            confidence_loss = self.confidence_loss(confidence, confidence_targets)
        
        # Combined loss
        total_loss = (
            self.lambda_macro * macro_loss
            + self.lambda_micro * micro_loss
            + self.lambda_center * center_loss
            + self.lambda_confidence * confidence_loss
        )
        
        return {
            "total_loss": total_loss,
            "macro_loss": macro_loss,
            "micro_loss": micro_loss,
            "center_loss": center_loss,
            "confidence_loss": confidence_loss,
        }
