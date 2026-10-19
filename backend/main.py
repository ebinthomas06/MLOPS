from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.webhook import router as webhook_router
from routes.repos import router as repos_router
from routes.status import router as status_router
from routes.pipeline import router as pipeline_router


app = FastAPI(
    title="AutoMLOps Backend",
    description="Backend API for the AutoMLOps platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhook_router)
app.include_router(repos_router)
app.include_router(status_router)
app.include_router(pipeline_router)


@app.get("/")
def root():
    return {
        "service": "AutoMLOps Backend",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
