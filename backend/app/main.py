from fastapi import FastAPI

from api.health import router as health_router


app = FastAPI(
    title="Bento API",
    version="0.1.0",
    description="Backend scaffold for multimodal relationship wellness workflows.",
)

app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Bento API", "status": "ready"}
