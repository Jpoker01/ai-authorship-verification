from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict

app = FastAPI(
    title="Authorship Verification DT - backend",
    description="API for verifying if two texts are written by the same author",
    version="1.0.0"
)

# CORS set up - only for demo purposes
# TODO: replace with domain once set to production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# included routers
app.include_router(predict.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": app.title,
        "version": app.version,
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}