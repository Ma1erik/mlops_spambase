import time
from collections import deque
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.service.schemas import (
    SpambaseItem,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
    SwitchVersionRequest,
)
from src.service.registry import ModelRegistry

REGISTRY_PATH = "reports/LAB6/registry.json"

app = FastAPI(title="Spambase Model Inference Service", version="1.0.0")

registry = ModelRegistry(registry_file=REGISTRY_PATH)

START_TIME = time.time()
latencies = deque(maxlen=1000)
request_counter = 0
prediction_counter = 0
error_counter = 0


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    global error_counter
    error_counter += 1
    safe_errors = []
    for err in exc.errors():
        err_copy = dict(err)
        if "ctx" in err_copy and isinstance(err_copy["ctx"], dict):
            err_copy["ctx"] = {k: str(v) for k, v in err_copy["ctx"].items()}
        safe_errors.append(err_copy)
    return JSONResponse(
        status_code=422, content={"detail": safe_errors, "message": "Validation error"}
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    active_ver, _ = registry.get_active_model()
    return HealthResponse(
        status="healthy",
        active_version=active_ver,
        uptime_seconds=round(time.time() - START_TIME, 2),
        loaded_models_count=len(registry._loaded_models),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(item: SpambaseItem):
    global request_counter, prediction_counter, error_counter
    t0 = time.perf_counter()
    request_counter += 1

    try:
        ver, model = registry.get_active_model()
        pred, prob = model.predict(item.features)
        prediction_counter += 1
        return PredictionResponse(
            prediction=pred, probability=round(prob, 4), model_version=ver, status="ok"
        )
    except Exception as e:
        error_counter += 1
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)


@app.get("/models")
async def list_models():
    return registry.registry_data


@app.post("/promote")
async def promote_model(payload: SwitchVersionRequest):
    if not payload.version:
        raise HTTPException(status_code=400, detail="Field 'version' is required")
    try:
        registry.promote(target_version=payload.version, reason=payload.reason)
        return {
            "status": "success",
            "active_version": registry.active_version,
            "message": f"Version {payload.version} activated",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rollback")
async def rollback_model(payload: SwitchVersionRequest):
    try:
        restored_ver = registry.rollback(reason=payload.reason)
        return {
            "status": "success",
            "active_version": restored_ver,
            "message": f"Rolled back to {restored_ver}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)

    return MetricsResponse(
        requests_total=request_counter,
        predictions_total=prediction_counter,
        errors_total=error_counter,
        active_version=registry.active_version or "unknown",
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        memory_rss_mb=round(rss_mb, 2),
        status="ok",
    )
