from fastapi import FastAPI
from app.config import Settings

app = FastAPI(title="sktime-agentic")
settings = Settings()

@app.get("/health")
def health_check():
    return {"status": "ok"}\n