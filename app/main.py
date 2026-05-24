import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import config
from app import ml_router, rag_router, integration_router
from src.rag.retrieve import ComplaintRetriever

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup lifespan hook loading the trained ML model package
    and loading/verifying the RAG index and metadata.
    """
    print("Executing FastAPI lifespan startup...")
    
    # 1. Load ML Model Package
    model_path = config.PROCESSED_DATA_DIR / "best_model.joblib"
    if model_path.exists():
        try:
            print(f"Loading best ML model from {model_path}...")
            app.state.model_package = joblib.load(model_path)
            print("ML model loaded successfully.")
        except Exception as e:
            print(f"Failed to load ML model package: {e}")
            app.state.model_package = None
    else:
        print("No deployment model package found. Run training pipeline to deploy.")
        app.state.model_package = None
        
    # 2. Check RAG Index Readiness
    retriever = ComplaintRetriever()
    try:
        retriever.load()
        app.state.rag_retriever = retriever
        print("RAG Index and metadata loaded successfully.")
    except Exception as e:
        print(f"RAG Index not ready/loaded: {e}")
        app.state.rag_retriever = None
        
    yield
    print("Shutdown complete.")

app = FastAPI(
    title="Meridian Customer Intelligence Platform API",
    description="Unified spine serving ML conversion prediction and grounded complaints Q&A RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ml_router.router)
app.include_router(rag_router.router)
app.include_router(integration_router.router)

@app.get("/health", tags=["System Health"])
def health_check():
    """
    Returns system liveness diagnostics and loaded model/index versions.
    """
    model_loaded = hasattr(app.state, "model_package") and app.state.model_package is not None
    rag_loaded = hasattr(app.state, "rag_retriever") and app.state.rag_retriever is not None
    
    model_version = "None"
    metrics = {}
    if model_loaded:
        model_version = app.state.model_package.get("run_id", "local_deploy")
        metrics = app.state.model_package.get("metrics", {})
        
    return {
        "status": "healthy",
        "services": {
            "ml_service": {
                "status": "loaded" if model_loaded else "not_loaded",
                "model_version": model_version,
                "metrics": metrics
            },
            "rag_service": {
                "status": "ready" if rag_loaded else "not_ready"
            }
        }
    }
