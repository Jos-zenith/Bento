.PHONY: run-backend

.PHONY: run-frontend

run-backend:
	uvicorn main:app --reload --app-dir backend/app

run-frontend:
	cd frontend && npm run dev


