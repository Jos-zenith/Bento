# Architecture

## Initial layout

- `backend/`: FastAPI service entrypoint and API contracts
- `frontend/`: web client placeholder
- `docs/`: product and architecture notes

## First milestones

1. Wire a `/health` endpoint into a running FastAPI app.
2. Add request/response schemas for text, audio, and video ingestion.
3. Introduce retrieval storage for mediation knowledge snippets.
4. Split CV, audio, and LLM workflows into separate services.
