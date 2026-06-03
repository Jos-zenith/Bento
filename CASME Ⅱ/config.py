# ============================================================================
# CASME II Micro-Expression Classification - Configuration
# ============================================================================

import torch
from typing import Optional

class Config:
    """Configuration for CASME II training pipeline"""
    
    # ===================== PATHS =====================
    DATA_ROOT = r"c:\Users\Zenith Joshua\CASME Ⅱ"
    RAW_DATA_PATH = r"c:\Users\Zenith Joshua\CASME Ⅱ\CASME2_RAW\CASME2-RAW"
    SELECTED_DATA_PATH = r"c:\Users\Zenith Joshua\CASME Ⅱ\CASME2_RAW_selected\CASME2_RAW_selected"
    COMPRESSED_VIDEO_PATH = r"c:\Users\Zenith Joshua\CASME Ⅱ\CASME2_Compressed video\CASME2_compressed"
    CROPPED_DATA_PATH = r"c:\Users\Zenith Joshua\CASME Ⅱ\Cropped\Cropped"

    DATA_STAGE = "cropped"  # raw | selected | compressed_video | cropped
    DATASET_ROOTS = {
        "raw": RAW_DATA_PATH,
        "selected": SELECTED_DATA_PATH,
        "compressed_video": COMPRESSED_VIDEO_PATH,
        "cropped": CROPPED_DATA_PATH,
    }

    @classmethod
    def get_data_root(cls, data_stage: Optional[str] = None) -> str:
        """Resolve the filesystem root for a CASME II stage."""
        stage = (data_stage or cls.DATA_STAGE).lower()
        return cls.DATASET_ROOTS.get(stage, cls.CROPPED_DATA_PATH)
    LABELS_FILE = r"c:\Users\Zenith Joshua\CASME Ⅱ\CASME2-coding-20140508.xlsx"
    
    # ===================== DATASET =====================
    NUM_SUBJECTS = 26
    FRAME_SIZE = (224, 224)  # EfficientNet-B0 input size
    TEMPORAL_LENGTH = 12  # Sample 12 frames from onset to apex
    NUM_CLASSES = 5  # Happiness, Surprise, Disgust, Repression, Others
    MODEL_INPUT_CHANNELS = 7  # RGB (3) + Combined Optical Flow (4)
    USE_COMBINED_OPTICAL_FLOW = True
    
    CLASS_MAPPING = {
        'Happiness': 0,
        'Surprise': 1,
        'Disgust': 2,
        'Repression': 3,
        'Others': 4
    }
    REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
    
    # ===================== MODEL =====================
    MODEL_TYPE = "3d_resnet_ms"  # Options: "3d_cnn", "3d_cnn_lstm", "3d_resnet_ms"
    BACKBONE = "efficientnet_b0"
    
    # 3D-CNN-LSTM Architecture
    NUM_3D_LAYERS = 2
    KERNEL_SIZE_3D = (3, 3, 3)
    PADDING_3D = (1, 1, 1)
    
    # LSTM settings
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # Fully connected layers
    FC_HIDDEN_DIMS = [512, 256]
    DROPOUT_RATE = 0.4
    
    # ===================== TRAINING =====================
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 5
    
    # Optimization
    OPTIMIZER = "adamw"  # Options: "adam", "adamw", "sgd"
    SCHEDULER = "cosine"  # Options: "cosine", "step", "linear"
    
    # Loss function
    LOSS_FN = "weighted_crossentropy"  # Account for class imbalance
    CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 1.2, 1.3, 0.8])  # Weighted by frequency
    
    # ===================== TRANSFER LEARNING =====================
    PRETRAINED_BACKBONE = True
    FREEZE_BACKBONE_EPOCHS = 10  # Freeze backbone for first N epochs
    FINETUNE_LR_RATIO = 0.1  # Backbone LR = main LR * ratio
    
    # ===================== DATA AUGMENTATION =====================
    USE_AUGMENTATION = True
    AUGMENTATION_PARAMS = {
        'random_flip': 0.3,
        'random_rotation': 15,
        'random_brightness': 0.2,
        'random_contrast': 0.2,
        'gaussian_noise_std': 0.01,
        'optical_flow_noise': 0.05,
    }
    
    # ===================== PREPROCESSING =====================
    NORMALIZE_OPTICAL_FLOW = True
    OPTICAL_FLOW_METHOD = "farneback"  # Options: "lucas_kanade", "farneback"
    CLIP_OPTICAL_FLOW = True
    FLOW_CLIPPING_VALUE = 20.0
    
    # Normalization stats (compute from training data)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    
    # ===================== VALIDATION & TESTING =====================
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.15
    TRAIN_SPLIT = 0.75
    
    # Cross-validation
    USE_K_FOLD = True
    K_FOLDS = 5
    
    # ===================== METRICS =====================
    TRACK_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
    LOG_CONFUSION_MATRIX = True
    
    # ===================== DEVICE =====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optimize data loading: small dataset benefits from fewer workers
    # With 193 training samples and batch_size=8, we only need 2-3 workers
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0
    # Pin memory only if GPU is available to prevent CPU-GPU contention
    PIN_MEMORY = torch.cuda.is_available()
    
    # ===================== CHECKPOINTING =====================
    CHECKPOINT_DIR = r"c:\Users\Zenith Joshua\CASME Ⅱ\checkpoints"
    BEST_MODEL_PATH = r"c:\Users\Zenith Joshua\CASME Ⅱ\checkpoints\best_model.pth"
    SAVE_FREQ = 5  # Save checkpoint every N epochs
    
    # ===================== LOGGING =====================
    LOG_DIR = r"c:\Users\Zenith Joshua\CASME Ⅱ\logs"
    TENSORBOARD = True
    LOG_INTERVAL = 10  # Log every N batches
    
    # ===================== WELLNESS SCORING =====================
    # Wellness Engagement Score: W(u) = α*A(u) + β*S(u) + γ*C(u)
    ALPHA = 0.5   # Affective/Micro-expression weight
    BETA = 0.3    # Sentiment weight
    GAMMA = 0.2   # Cultural alignment weight
    
    # Emotion valence scores (affects wellness)
    EMOTION_VALENCE = {
        'Happiness': 1.0,      # Positive
        'Surprise': 0.0,       # Neutral
        'Disgust': -0.8,       # Negative
        'Repression': -0.5,    # Negative (suppressed)
        'Others': 0.0          # Neutral
    }


# ===================== RUNTIME CONFIGS =====================

class TrainConfig(Config):
    """Training-specific configurations"""
    MODE = "train"
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    

class EvalConfig(Config):
    """Evaluation-specific configurations"""
    MODE = "eval"
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    

class InferenceConfig(Config):
    """Inference-specific configurations"""
    MODE = "inference"
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Export
cfg = Config()
train_cfg = TrainConfig()
eval_cfg = EvalConfig()
infer_cfg = InferenceConfig()
