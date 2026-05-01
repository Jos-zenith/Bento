# Facial Emotion Recognition Module

## Overview

This module provides **production-grade facial emotion recognition** using EfficientNet-B0 backbone fine-tuned on macro-expressions (FER2013) and micro-expressions (CASME II, SAMM).

The system detects **involuntary emotional leaks** - micro-expressions that reveal genuine emotions despite intentional suppression, critical for relationship wellness analysis.

## Architecture

```
EfficientNet-B0 Backbone (Pre-trained)
    ↓
Feature Extraction (1280 features)
    ↓
[Spatial Attention Module] (optional)
    ↓
Shared Embedding Layer (512-dim)
    ├→ Macro-Expression Head (7 classes: FER2013)
    │  └→ Happy, Sad, Angry, Fearful, Disgusted, Surprised, Neutral
    │
    ├→ Micro-Expression Head (5 classes: CASME II/SAMM)
    │  └→ Positive, Negative, Surprise, Repression, Others
    │
    └→ Confidence Head
       └→ Detection reliability score
```

## Key Features

### 1. Dual Expression Detection
- **Macro-Expressions**: Intentional or easily detectable emotions (0.5-4 seconds)
- **Micro-Expressions**: Involuntary emotional leaks (1/25 - 1/3 second)

### 2. Multi-Dataset Fine-tuning
- **FER2013**: 35,000+ macro-expression images, 7 emotion classes
- **CASME II**: 255 micro-expressions, spontaneous emotional reactions
- **SAMM**: 159 micro-expressions, high-speed (200 fps) capture

### 3. Advanced Loss Functions
- **Focal Loss**: Handles micro-expression rarity and class imbalance
- **Center Loss**: Improves embedding discrimination
- **Confidence Loss**: Trains detection reliability

### 4. Spatial Attention
Focuses network on emotion-relevant facial regions (eyes, mouth, brows)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or via pyproject.toml
pip install -e .
```

### Requirements
- PyTorch ≥ 2.0.0
- TorchVision ≥ 0.15.0
- OpenCV ≥ 4.8.0
- NumPy ≥ 1.24.0

## Dataset Setup

### FER2013
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── ... (same structure)
```

Download: [Kaggle FER2013](https://www.kaggle.com/datasets/murngl/facial-expression-recognition-dataset)

### CASME II
```
casme_ii/
├── train/
│   └── subject*/
│       └── video*/
│           ├── 0001.jpg
│           ├── 0002.jpg
│           └── annotation.txt
└── test/
```

Download: [CASME II](http://fu.psych.ac.cn/CASME/casme2-en.html)

### SAMM
```
samm/
└── subject*/
    └── video*/
        ├── 01_0001.jpg
        ├── 01_0002.jpg
        └── annotation.txt
```

Download: [SAMM](http://www2.docm.mmu.ac.uk/SAMM/)

## Usage

### 1. Training

#### Single Dataset (FER2013)
```bash
python affective_intelligence/train.py \
    --fer2013_path ./data/FER2013 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3
```

#### Multi-Dataset (All three)
```bash
python affective_intelligence/train.py \
    --fer2013_path ./data/FER2013 \
    --casme_path ./data/CASME_II \
    --samm_path ./data/SAMM \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --output_dir ./checkpoints
```

#### Advanced Options
```bash
python affective_intelligence/train.py \
    --fer2013_path ./data/FER2013 \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --embedding_dim 512 \
    --dropout_rate 0.3 \
    --use_attention
```

### 2. Inference

#### Python API
```python
from affective_intelligence.inference import EmotionPredictor
import cv2

# Initialize predictor
predictor = EmotionPredictor(
    model_path="./checkpoints/best_model.pt",
    device="cuda"
)

# Load image
image = cv2.imread("face.jpg")

# Predict macro-expression
macro_result = predictor.predict_macro_emotion(image)
print(f"Macro: {macro_result['emotion']} ({macro_result['confidence']:.2%})")

# Predict micro-expression
micro_result = predictor.predict_micro_emotion(image, confidence_threshold=0.5)
print(f"Micro: {micro_result['emotion']}")
print(f"Is genuine micro-expression: {micro_result['is_micro_expression']}")

# Get both predictions
both = predictor.predict_both(image)
```

#### FastAPI REST Endpoints

**Macro-Expression Prediction**
```bash
curl -X POST "http://localhost:8000/api/v1/emotion/predict/macro" \
  -F "file=@face.jpg"
```

Response:
```json
{
  "emotion": "happy",
  "confidence": 0.92,
  "class_scores": {
    "angry": 0.01,
    "disgust": 0.02,
    "fear": 0.00,
    "happy": 0.92,
    "neutral": 0.03,
    "sad": 0.01,
    "surprise": 0.01
  },
  "type": "macro"
}
```

**Micro-Expression Prediction**
```bash
curl -X POST "http://localhost:8000/api/v1/emotion/predict/micro" \
  -F "file=@face.jpg" \
  -F "confidence_threshold=0.5"
```

Response:
```json
{
  "emotion": "positive",
  "confidence": 0.78,
  "class_scores": {
    "positive": 0.78,
    "negative": 0.10,
    "surprise": 0.05,
    "repression": 0.05,
    "others": 0.02
  },
  "type": "micro"
}
```

**Dual Prediction** (Macro + Micro)
```bash
curl -X POST "http://localhost:8000/api/v1/emotion/predict/dual" \
  -F "file=@face.jpg"
```

Response:
```json
{
  "macro_emotion": "happy",
  "macro_confidence": 0.92,
  "micro_emotion": "positive",
  "micro_confidence": 0.78,
  "is_micro_expression": true,
  "micro_detection_confidence": 0.73
}
```

**Model Info**
```bash
curl "http://localhost:8000/api/v1/emotion/model/info"
```

### 3. Detailed Examples

#### Video Stream Processing
```python
import cv2
from affective_intelligence.inference import EmotionPredictor

predictor = EmotionPredictor("./checkpoints/best_model.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect emotions
    result = predictor.predict_both(frame)
    
    # Visualize macro-expression
    macro_emotion = result["macro"]["emotion"]
    macro_conf = result["macro"]["confidence"]
    cv2.putText(frame, f"Macro: {macro_emotion} ({macro_conf:.2%})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Visualize micro-expression if detected
    if result["micro"]["is_micro_expression"]:
        micro_emotion = result["micro"]["emotion"]
        cv2.putText(frame, f"MICRO: {micro_emotion} (involuntary)", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Batch Processing
```python
from pathlib import Path
import pandas as pd
from affective_intelligence.inference import EmotionPredictor
import cv2

predictor = EmotionPredictor("./checkpoints/best_model.pt")

results = []
for img_path in Path("./images").glob("*.jpg"):
    image = cv2.imread(str(img_path))
    pred = predictor.predict_both(image)
    
    results.append({
        "image": img_path.name,
        "macro_emotion": pred["macro"]["emotion"],
        "macro_conf": pred["macro"]["confidence"],
        "micro_emotion": pred["micro"]["emotion"],
        "micro_conf": pred["micro"]["classification_confidence"],
        "is_micro_expr": pred["micro"]["is_micro_expression"],
    })

df = pd.DataFrame(results)
df.to_csv("emotion_results.csv", index=False)
```

## Model Architecture Details

### EmotionNet Components

1. **EfficientNet-B0 Backbone**
   - Pre-trained on ImageNet
   - Efficient: ~5.3M parameters
   - Output: 1280-dim feature maps

2. **Spatial Attention Module**
   - Channel attention via FC layers
   - Focuses on emotion-relevant features
   - Improves micro-expression detection

3. **Shared Embedding Layer**
   - 1280 → 512 dimensions
   - BatchNorm + ReLU + Dropout
   - Learned representation space

4. **Classification Heads**
   - Separate for macro (7 classes) and micro (5 classes)
   - 512 → 256 → output dims
   - Independent optimization

5. **Confidence Head**
   - Predicts detection reliability
   - Sigmoid output: [0, 1]
   - Indicates involuntary emotional leak probability

### Loss Functions

#### Focal Loss
```
FL = -α_t * (1 - p_t)^γ * log(p_t)
```
- Focuses on hard examples
- Handles class imbalance in micro-expressions

#### Center Loss
```
L_c = ||embedding - center||²
```
- Pulls embeddings toward class centers
- Improves discrimination

#### Combined Loss
```
L_total = λ_macro * CE(macro) + λ_micro * FL(micro) + λ_center * L_c + λ_conf * BCE(confidence)
```

## Performance Metrics

### Training Configuration
```yaml
Model: EfficientNet-B0 (5.3M params)
Epochs: 100
Batch Size: 32
Learning Rate: 1e-3 (cosine annealing)
Optimizer: AdamW
Loss: Combined (macro + focal + center + confidence)
Data: FER2013 + CASME II + SAMM
```

### Expected Results
- **Macro-Expression**: ~70-75% accuracy on FER2013 test set
- **Micro-Expression**: ~60-65% accuracy on CASME II/SAMM
- **Micro Detection**: ~80%+ confidence calibration

Note: Micro-expression recognition is inherently challenging due to:
- Temporal nature (rapid movements)
- Subtle facial changes
- High inter-subject variability
- Limited dataset sizes

## Configuration

### Environment Variables
```bash
# Model path for API
EMOTION_MODEL_PATH=./models/emotion_model.pt

# Device
CUDA_VISIBLE_DEVICES=0
```

### EmotionNetConfig
```python
from affective_intelligence.models import EmotionNetConfig

config = EmotionNetConfig(
    num_macro_classes=7,
    num_micro_classes=5,
    pretrained=True,
    dropout_rate=0.3,
    use_attention=True,
    embedding_dim=512,
)
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Or use CPU
--device cpu
```

### Poor Micro-Expression Accuracy
- Ensure CASME II/SAMM frames are properly aligned
- Increase training epochs
- Use higher learning rate: `--learning_rate 5e-4`

### Model Loading Error
```python
# Check checkpoint format
import torch
ckpt = torch.load("model.pt")
print(ckpt.keys())  # Should have 'model_state_dict'
```

## References

1. **EfficientNet**: Tan & Le (2019) - https://arxiv.org/abs/1905.11946
2. **Focal Loss**: Lin et al. (2017) - https://arxiv.org/abs/1708.02002
3. **Center Loss**: Wen et al. (2016) - https://arxiv.org/abs/1612.00134
4. **FER2013**: Goodfellow et al. (2013) - https://arxiv.org/abs/1307.0414
5. **CASME II**: Qu et al. (2014) - Database description
6. **SAMM**: Davison et al. (2016) - Database description

## Citation

If you use this module, please cite:

```bibtex
@software{panjaayathu_emotion_2025,
  title={Facial Emotion Recognition Module for Panjaayathu},
  author={Affective Intelligence Team},
  year={2025},
  note={EfficientNet-B0 based macro and micro-expression detection}
}
```

## License

Part of the Panjaayathu project. See main repository LICENSE.

## Support

For issues, questions, or improvements:
1. Check this documentation
2. Review example notebooks
3. Submit issues to repository
