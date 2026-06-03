import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.health import router as health_router
from api.emotion import router as emotion_router, init_emotion_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown events."""
    # Startup
    model_path = os.getenv("EMOTION_MODEL_PATH", "./models/emotion_model.pt")
    if os.path.exists(model_path):
        try:
            init_emotion_predictor(model_path)
            print(f"✓ Emotion recognition model loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load emotion model: {e}")
    else:
        print(f"⚠ Emotion model not found at {model_path}. Emotion endpoints may not work.")
    
    yield
    
    # Shutdown
    print("Shutting down Bento API...")


app = FastAPI(
    title="Bento API",
    version="0.1.0",
    description="Backend scaffold for multimodal relationship wellness workflows.",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(emotion_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Bento API", "status": "ready"}
