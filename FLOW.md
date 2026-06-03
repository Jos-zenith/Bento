# Bento System Flow & Architecture

## Project Overview
**Bento** is a production-grade multimodal relationship wellness platform that combines PyTorch computer vision, LLMs, and culturally-aware reasoning to analyze user interactions and provide mediation guidance.

---

## System Architecture Layers

### 1. **Frontend (Next.js)**
- **Location:** `/frontend`
- **Purpose:** Modern web interface for demos and user interaction
- **Features:**
  - Landing page showcasing architecture
  - Integration with backend API
  - Responsive design with modern UI

**Key Files:**
- `app/page.tsx` - Main landing page with flow visualization
- `next.config.ts` - Next.js configuration
- `package.json` - Dependencies

---

### 2. **Backend (FastAPI)**
- **Location:** `/backend/app`
- **Purpose:** REST API server orchestrating all AI pipelines

**Key Components:**

#### a. **Core FastAPI Application** (`main.py`)
- Application initialization with lifespan management
- Router registration (health, emotion endpoints)
- Model loading on startup
- Service health endpoint

#### b. **API Endpoints** (`app/api/`)
- **Health Router** (`health.py`) - Service status checks
- **Emotion Router** (`emotion.py`) - Emotion prediction endpoints
  - `/api/v1/emotion/predict/macro` - Macro-expression analysis
  - `/api/v1/emotion/predict/micro` - Micro-expression analysis

#### c. **Core Modules** (`app/core/`)
- Utility functions and shared configurations

---

### 3. **AI Engine (PyTorch)**
- **Location:** `/backend/affective_intelligence`
- **Purpose:** Computer vision and emotion recognition models

**Key Components:**

#### a. **Models** (`models/emotion_net.py`)
- EfficientNet-V2 based emotion classification
- Fine-tuned for both macro and micro-expressions
- Handles 7+ emotion classes

#### b. **Inference** (`inference/emotion_predictor.py`)
- EmotionPredictor class for runtime predictions
- Model loading and inference orchestration
- Confidence scoring

#### c. **Datasets** (`datasets/`)
- `fer2013.py` - Facial Expression Recognition 2013 dataset integration
- `micro_expressions.py` - Micro-expression dataset handling
- `transforms.py` - Image preprocessing and augmentation

#### d. **Training** (`train.py`)
- Model training pipeline
- Loss functions and optimization

#### e. **Losses** (`losses/emotion_losses.py`)
- Custom loss functions for emotion recognition

---

### 4. **Macro-Expression Analysis**
- **Location:** `/CASME Ⅱ`
- **Purpose:** Micro-expression specific analysis pipeline
- **Features:**
  - Optical flow preprocessing
  - Video frame extraction and analysis
  - Temporal understanding of expressions

**Key Files:**
- `train.py` - Training pipeline
- `inference.py` - Inference execution
- `models.py` - Model definitions
- `dataset.py` - Data loading
- `preprocessing.py` - Frame extraction and normalization

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                           │
│                    (Frontend / Next.js)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │     FastAPI Backend (main.py)     │
         │   - Request routing               │
         │   - Model lifecycle management    │
         └──────────┬────────────────────────┘
                    │
         ┌──────────▼────────────────────────┐
         │      API Endpoints                 │
         │  /api/v1/emotion/predict/*        │
         └──────────┬────────────────────────┘
                    │
     ┌──────────────┼──────────────┐
     │              │              │
     ▼              ▼              ▼
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │  Image   │  │  Video   │  │  Audio   │
 │  Input   │  │  Input   │  │  Input   │
 └────┬─────┘  └────┬─────┘  └────┬─────┘
      │             │             │
      ▼             ▼             ▼
 ┌────────────────────────────────────────┐
 │    PyTorch AI Engine                   │
 │  (affective_intelligence)              │
 │                                        │
 │  - Vision: CV layer for emotion        │
 │  - Inference: EmotionPredictor         │
 │  - Models: EfficientNet-V2             │
 │  - Data: FER2013, Micro-expressions    │
 └────┬───────────────────────────────────┘
      │
      ├─────────────────────────────────┐
      │                                 │
      ▼                                 ▼
 ┌──────────────────┐           ┌─────────────────┐
 │ Emotion Classes  │           │ Confidence      │
 │ (Happy, Sad,     │           │ Scores          │
 │  Angry, etc.)    │           │ (0.0-1.0)       │
 └────┬─────────────┘           └────────┬────────┘
      │                                  │
      └──────────────┬───────────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │  Response JSON         │
          │  - Emotion             │
          │  - Confidence          │
          │  - Class scores        │
          │  - Type (macro/micro)  │
          └────────┬───────────────┘
                   │
                   ▼
          ┌──────────────────────┐
          │ Frontend Display     │
          │ - Visual feedback    │
          │ - Confidence metrics │
          │ - Recommendations    │
          └──────────────────────┘
```

---

## Request Flow Breakdown

### Step 1: User Input
- User uploads video, image, or text via frontend
- Data sent to backend API endpoint

### Step 2: Validation
- FastAPI validates request format
- File type and size checks

### Step 3: Preprocessing
- Image extraction from video (if applicable)
- Normalization and augmentation via `transforms.py`
- Face detection and landmark extraction (MediaPipe/MTCNN)

### Step 4: Emotion Recognition
- Input passed to `EmotionPredictor`
- Model inference via PyTorch
- Confidence scoring

### Step 5: Response Generation
- Results formatted as JSON response
- Emotion class + confidence score
- Categorization (macro vs micro)

### Step 6: Frontend Display
- JSON response rendered in UI
- Visual feedback and recommendations displayed

---

## Service Startup Flow

```
1. Backend Startup (main.py lifespan context)
   ├─ Load environment variables
   ├─ Initialize EmotionPredictor
   ├─ Load PyTorch model from disk
   ├─ Verify GPU availability (if applicable)
   └─ Ready to serve requests

2. Frontend Startup (Next.js)
   ├─ Compile TypeScript/React
   ├─ Load assets
   ├─ Establish API connection to backend
   └─ Display landing page

3. Request Handling
   ├─ Accept multimodal input (image/video/audio)
   ├─ Route to appropriate handler
   ├─ Execute model inference
   └─ Return structured response
```

---

## Key Data Structures

### EmotionPredictionResponse
```json
{
  "emotion": "happy",
  "confidence": 0.95,
  "class_scores": {
    "happy": 0.95,
    "sad": 0.02,
    "angry": 0.01,
    "neutral": 0.02
  },
  "type": "macro"
}
```

### DualEmotionPredictionResponse
```json
{
  "macro_emotion": "neutral",
  "macro_confidence": 0.87,
  "micro_emotion": "fear",
  "micro_confidence": 0.92,
  "is_micro_expression": true,
  "micro_detection_confidence": 0.88
}
```

---

## Environment Configuration

**Backend Variables:**
- `EMOTION_MODEL_PATH` - Path to PyTorch emotion model (default: `./models/emotion_model.pt`)
- `CUDA_VISIBLE_DEVICES` - GPU selection for PyTorch

**Frontend Variables:**
- `NEXT_PUBLIC_API_URL` - Backend API base URL (default: `http://127.0.0.1:8000`)

---

## Development Commands

### Backend
```bash
# Start FastAPI server with hot reload
make run-backend
# Or: uvicorn main:app --reload --app-dir backend/app
```

### Frontend
```bash
# Start Next.js dev server
make run-frontend
# Or: cd frontend && npm run dev
```

### Verify Setup
```bash
# Run setup verification
python verify_setup.py
```

---

## Training Pipeline

### Emotion Model Training
1. **Data Loading:** FER2013 + custom micro-expression datasets
2. **Preprocessing:** Image normalization, augmentation via `transforms.py`
3. **Model:** EfficientNet-V2 backbone with emotion-specific head
4. **Training:** `backend/train.py` or `CASME Ⅱ/train.py`
5. **Validation:** Cross-validation on held-out test set
6. **Export:** Save as PyTorch checkpoint (`.pt` file)

### Micro-Expression Pipeline
1. **Video Preprocessing:** Optical flow computation
2. **Frame Extraction:** Key frame selection from video
3. **Model Inference:** Temporal understanding via VideoMAE
4. **Classification:** Micro vs macro distinction
5. **Temporal Analysis:** Motion tracking across frames

---

## Current Status

✅ **Implemented:**
- FastAPI backend with emotion API
- Next.js landing page with modern UI
- PyTorch model infrastructure
- Macro and micro-expression detection framework

⚠️ **In Development:**
- Model training and checkpoint loading
- VLM and RAG integration
- Audio emotion recognition (Wav2Vec2/HuBERT)
- LangGraph reasoning layer
- Vernacular LLM integration

---

## Next Steps

1. **Model Training:** Train emotion recognition models on FER2013 + micro-expression datasets
2. **API Integration:** Wire up inference endpoints with trained models
3. **Frontend Enhancement:** Add video upload and real-time analysis UI
4. **Testing:** Comprehensive test suite for API and models
5. **Optimization:** TensorRT optimization for production latency
6. **Deployment:** Containerization and cloud deployment pipeline

