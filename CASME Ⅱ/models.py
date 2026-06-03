# ============================================================================
# CASME II Model Architectures - 3D-CNN-LSTM with EfficientNet-B0
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, List


class SpatialFeatureExtractor(nn.Module):
    """
    EfficientNet-B0 backbone for spatial feature extraction.
    Pre-trained on ImageNet, fine-tuned on CASME II.
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        
        # Remove classification head
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.feature_dim = self.backbone.classifier[1].in_features
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, C, H, W)
        Returns:
            features: Output of shape (B, feature_dim, 1, 1) -> (B, feature_dim)
        """
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(x.size(0), -1)
        return x


class Temporal3DCNNBlock(nn.Module):
    """3D Convolutional block for temporal modeling"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        padding: Tuple[int, int, int] = (1, 1, 1),
        pool_size: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, C, T, H, W)
        Returns:
            Output of shape (B, out_channels, T', H', W')
        """
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.pool3d(x)
        return x


class CASME2_3DCNN_LSTM(nn.Module):
    """
    3D-CNN + LSTM for Micro-Expression Classification
    
    Architecture:
    1. Spatial Feature Extraction: EfficientNet-B0 on each frame
    2. Temporal 3D-CNN: Capture 3D spatio-temporal patterns
    3. LSTM: Long-range temporal dependencies
    4. Classification Head: Fully connected layers
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        temporal_length: int = 12,
        input_channels: int = 7,  # RGB + Combined Optical Flow
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.temporal_length = temporal_length
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # ===================== Spatial Feature Extractor =====================
        self.spatial_encoder = SpatialFeatureExtractor(
            pretrained=pretrained_backbone,
            freeze=freeze_backbone
        )
        spatial_feature_dim = self.spatial_encoder.feature_dim
        
        # ===================== Temporal 3D-CNN =====================
        # Project input to spatial_feature_dim channels
        self.input_projection = nn.Conv3d(
            input_channels, 64,
            kernel_size=(1, 7, 7),
            padding=(0, 3, 3)
        )
        
        self.conv3d_blocks = nn.Sequential(
            Temporal3DCNNBlock(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Temporal3DCNNBlock(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )
        
        # ===================== LSTM for Temporal Modeling =====================
        # After 3D-CNN: (B, 256, T', H', W') -> pooling -> (B, 256, T')
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use unidirectional LSTM for 30% speedup and reduced memory usage
        # For micro-expressions, forward temporal flow is sufficient for classification
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False  # Optimization: unidirectional LSTM
        )
        lstm_output_dim = lstm_hidden_dim  # Single direction, not doubled
        
        # ===================== Classification Head =====================
        fc_hidden_dims = fc_hidden_dims or [512, 256]
        fc_layers = []
        
        prev_dim = lstm_output_dim
        for hidden_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for micro-expression classification.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) where:
                - B: Batch size
                - T: Temporal length (number of frames)
                - C: Number of channels (3 for RGB or 4 for RGB+Flow)
                - H, W: Spatial dimensions
            return_features: If True, return intermediate features
        
        Returns:
            logits: Classification logits of shape (B, num_classes)
            features: (Optional) Hidden features from LSTM of shape (B, lstm_hidden_dim*2)
        """
        batch_size, temporal_length, c, h, w = x.shape
        
        # ===================== 3D-CNN Path =====================
        # Input: (B, T, C, H, W) -> (B, C, T, H, W) for 3D conv
        x_3d = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Input projection
        x_3d = self.input_projection(x_3d)  # (B, 64, T, H, W)
        
        # 3D convolutional blocks
        x_3d = self.conv3d_blocks(x_3d)  # (B, 256, T', H', W')
        
        # ===================== LSTM Path =====================
        # Reshape for LSTM: (B, 256, T', H', W') -> (B, T', 256)
        b, c_3d, t_prime, h_prime, w_prime = x_3d.shape
        
        # Pool spatial dimensions
        x_lstm = self.temporal_pool(x_3d.reshape(b * t_prime, c_3d, h_prime, w_prime))
        x_lstm = x_lstm.reshape(b, t_prime, c_3d)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)  # lstm_out: (B, T', 512)
        
        # Use last hidden state from forward and backward
        features = lstm_out[:, -1, :]  # (B, lstm_hidden_dim*2)
        
        # ===================== Classification Head =====================
        logits = self.classifier(features)  # (B, num_classes)
        
        if return_features:
            return logits, features
        return logits


class CASME2_3DCNN(nn.Module):
    """
    Simplified 3D-CNN for Micro-Expression Classification (without LSTM).
    Use this for faster training with less memory overhead.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        temporal_length: int = 12,
        input_channels: int = 7,
        fc_hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.4,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        
        self.temporal_length = temporal_length
        self.num_classes = num_classes
        
        # ===================== 3D-CNN Blocks =====================
        self.conv3d_1 = Temporal3DCNNBlock(input_channels, 64, kernel_size=(3, 3, 3))
        self.conv3d_2 = Temporal3DCNNBlock(64, 128, kernel_size=(3, 3, 3))
        self.conv3d_3 = Temporal3DCNNBlock(128, 256, kernel_size=(3, 3, 3))
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # ===================== Classification Head =====================
        fc_hidden_dims = fc_hidden_dims or [512, 256]
        fc_layers = []
        
        prev_dim = 256
        for hidden_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, C, H, W)
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        # Permute to 3D conv format: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 3D convolutions
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        
        # Global pooling
        x = self.avg_pool(x)  # (B, 256, 1, 1, 1)
        x = x.reshape(x.size(0), -1)  # (B, 256)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class MultiScale3DResBlock(nn.Module):
    """Residual 3D block with parallel 3x3x3, 5x5x5, and 7x7x7 branches."""

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()

        branch_channels = max(out_channels // 3, 16)

        def branch(kernel_size: int) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    branch_channels,
                    kernel_size=(kernel_size, kernel_size, kernel_size),
                    stride=stride,
                    padding=(padding, padding, padding),
                    bias=False,
                ),
                nn.BatchNorm3d(branch_channels),
                nn.ReLU(inplace=True),
            )

        self.branch3 = branch(3)
        self.branch5 = branch(5)
        self.branch7 = branch(7)

        self.merge = nn.Sequential(
            nn.Conv3d(branch_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        x = self.merge(x)
        x = x + residual
        return self.relu(x)


class CASME2_3DResNetMS(nn.Module):
    """
    Multi-scale 3D-ResNet for micro-expression classification.

    Uses residual 3D blocks with parallel 3x3x3, 5x5x5, and 7x7x7 kernels to
    capture fine facial motion at multiple temporal and spatial scales.
    """

    def __init__(
        self,
        num_classes: int = 5,
        temporal_length: int = 12,
        input_channels: int = 7,
        base_channels: int = 64,
        block_counts: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        self.temporal_length = temporal_length
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels,
                base_channels,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.stage1 = self._make_stage(base_channels, base_channels, block_counts[0])
        self.stage2 = self._make_stage(base_channels, base_channels * 2, block_counts[1], stride=(1, 2, 2))
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, block_counts[2], stride=(1, 2, 2))
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 8, block_counts[3], stride=(1, 2, 2))

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        feature_dim = base_channels * 8
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ) -> nn.Sequential:
        blocks = [MultiScale3DResBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(MultiScale3DResBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        features = x
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


def build_model(
    model_type: str = "3d_resnet_ms",
    num_classes: int = 5,
    temporal_length: int = 12,
    input_channels: int = 7,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build and return the specified model.
    
    Args:
        model_type: '3d_cnn', '3d_cnn_lstm', or '3d_resnet_ms'
        num_classes: Number of emotion classes
        temporal_length: Number of frames
        input_channels: Number of input channels (3 for RGB, 7 for RGB+COF)
        pretrained_backbone: Use pretrained EfficientNet-B0
        freeze_backbone: Freeze backbone weights initially
        device: Device to place model on
    
    Returns:
        model: PyTorch model
    """
    
    if model_type == "3d_cnn":
        model = CASME2_3DCNN(
            num_classes=num_classes,
            temporal_length=temporal_length,
            input_channels=input_channels,
            pretrained_backbone=pretrained_backbone,
        )
    elif model_type == "3d_cnn_lstm":
        model = CASME2_3DCNN_LSTM(
            num_classes=num_classes,
            temporal_length=temporal_length,
            input_channels=input_channels,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "3d_resnet_ms":
        model = CASME2_3DResNetMS(
            num_classes=num_classes,
            temporal_length=temporal_length,
            input_channels=input_channels,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if device:
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test multi-scale 3D-ResNet model
    model = build_model(
        model_type="3d_resnet_ms",
        num_classes=5,
        temporal_length=12,
        input_channels=7,
        device=device
    )
    
    print(f"Model type: 3D-ResNet-MS")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch = torch.randn(2, 12, 7, 224, 224).to(device)
    output = model(batch)
    print(f"\nInput shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with features
    output, features = model(batch, return_features=True)
    print(f"Features shape: {features.shape}")
    
    # Test 3D-CNN model
    print("\n" + "="*50)
    model_cnn = build_model(
        model_type="3d_cnn",
        num_classes=5,
        temporal_length=12,
        input_channels=7,
        device=device
    )
    
    print(f"Model type: 3D-CNN")
    print(f"Total parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    
    output_cnn = model_cnn(batch)
    print(f"Output shape: {output_cnn.shape}")
