import joblib
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src import config
from app import ml_router, rag_router, integration_router
from src.rag.retrieve import ComplaintRetriever

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup lifespan hook loading the trained ML model package,
    initializing in-memory metrics state, and loading/verifying the RAG index.
    """
    print("Executing FastAPI lifespan startup...")
    
    # 1. Initialize Observability Metrics
    app.state.metrics = {
        "requests_total": 0,
        "requests_by_endpoint": {
            "/ml/predict": 0,
            "/ml/batch-score": 0,
            "/rag/ask-complaints": 0,
            "/integration/customer-intel": 0,
            "/health": 0
        },
        "errors_total": 0,
        "errors_by_endpoint": {
            "/ml/predict": 0,
            "/ml/batch-score": 0,
            "/rag/ask-complaints": 0,
            "/integration/customer-intel": 0,
            "/health": 0
        },
        "latency_sum": 0.0,
        "latency_count": 0,
        "prediction_distribution": {
            "High Priority": 0,
            "Medium Priority": 0,
            "Low Priority": 0
        },
        "rag_retrieval_stats": {
            "total_queries": 0,
            "total_refusals": 0,
            "total_evidence_ids_retrieved": 0,
            "similarity_scores_sum": 0.0,
            "similarity_scores_count": 0
        }
    }
    
    # 2. Load ML Model Package
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
        
    # 3. Check RAG Index Readiness
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

# Custom Http Observability Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    is_api_path = path in [
        "/ml/predict",
        "/ml/batch-score",
        "/rag/ask-complaints",
        "/integration/customer-intel",
        "/health"
    ]
    
    if not is_api_path or not hasattr(app.state, "metrics"):
        return await call_next(request)
        
    # Increment total and endpoint-specific request counts
    app.state.metrics["requests_total"] += 1
    app.state.metrics["requests_by_endpoint"][path] += 1
    
    start_time = time.time()
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            app.state.metrics["errors_total"] += 1
            app.state.metrics["errors_by_endpoint"][path] += 1
        return response
    except Exception as e:
        app.state.metrics["errors_total"] += 1
        app.state.metrics["errors_by_endpoint"][path] += 1
        raise e
    finally:
        # Measure latency in milliseconds
        duration_ms = (time.time() - start_time) * 1000.0
        app.state.metrics["latency_sum"] += duration_ms
        app.state.metrics["latency_count"] += 1

# Register routers
app.include_router(ml_router.router)
app.include_router(rag_router.router)
app.include_router(integration_router.router)

@app.get("/health", tags=["System Health"])
def health_check():
    """
    Returns system liveness diagnostics, model version, and vector index version.
    """
    model_loaded = hasattr(app.state, "model_package") and app.state.model_package is not None
    rag_loaded = hasattr(app.state, "rag_retriever") and app.state.rag_retriever is not None
    
    model_version = "None"
    metrics = {}
    if model_loaded:
        model_version = app.state.model_package.get("run_id", "local_deploy")
        metrics = app.state.model_package.get("metrics", {})
        
    vector_index_version = "None"
    if rag_loaded:
        vector_index_version = "v1.0.0"
        
    return {
        "status": "healthy",
        "model_version": model_version,
        "vector_index_version": vector_index_version,
        "services": {
            "ml_service": {
                "status": "loaded" if model_loaded else "not_loaded",
                "model_version": model_version,
                "metrics": metrics
            },
            "rag_service": {
                "status": "ready" if rag_loaded else "not_ready",
                "vector_index_version": vector_index_version
            }
        }
    }

@app.get("/metrics", tags=["System Observability"])
def get_observability_metrics():
    """
    Surfaces core system performance statistics, including request distributions,
    error counts, prediction priority bands, and RAG semantic retrieval metrics.
    """
    metrics = getattr(app.state, "metrics", {})
    
    # Calculate average request latency
    avg_latency_ms = 0.0
    if metrics.get("latency_count", 0) > 0:
        avg_latency_ms = round(metrics["latency_sum"] / metrics["latency_count"], 2)
        
    # Calculate RAG average similarity score
    rag = metrics.get("rag_retrieval_stats", {})
    avg_similarity = 0.0
    if rag.get("similarity_scores_count", 0) > 0:
        avg_similarity = round(rag["similarity_scores_sum"] / rag["similarity_scores_count"], 4)
        
    total_reqs = metrics.get("requests_total", 0)
    success_rate = 1.0
    if total_reqs > 0:
        success_rate = round((total_reqs - metrics.get("errors_total", 0)) / total_reqs, 4)
        
    return {
        "status": "healthy",
        "observability": {
            "requests_total": total_reqs,
            "success_rate": success_rate,
            "errors_total": metrics.get("errors_total", 0),
            "average_latency_ms": avg_latency_ms,
            "requests_by_endpoint": metrics.get("requests_by_endpoint", {}),
            "errors_by_endpoint": metrics.get("errors_by_endpoint", {})
        },
        "model_prediction_distribution": metrics.get("prediction_distribution", {}),
        "rag_retrieval_metrics": {
            "total_queries": rag.get("total_queries", 0),
            "total_refusals": rag.get("total_refusals", 0),
            "total_evidence_ids_retrieved": rag.get("total_evidence_ids_retrieved", 0),
            "average_similarity_score": avg_similarity
        }
    }
