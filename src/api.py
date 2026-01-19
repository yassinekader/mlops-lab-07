from __future__ import annotations

"""
API FastAPI de prédiction de churn pour le lab MLOps.

Ce service :
- charge dynamiquement le modèle actif depuis MLflow Model Registry (alias production) ;
- expose un endpoint `/health` pour vérifier l'état de l'API et du modèle ;
- expose un endpoint `/predict` pour faire une prédiction de churn ;
- journalise chaque requête de prédiction dans `logs/predictions.log` au format JSON.

Cette API illustre une étape "Serve" dans un pipeline MLOps minimal :
un modèle versionné est promu côté MLflow, puis utilisé par un service d’inférence.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constantes de chemin
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parents[1]
LOG_PATH: Path = ROOT / "logs" / "predictions.log"


# ---------------------------------------------------------------------------
# Constantes MLflow (Model Registry)
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "churn_model"
ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="MLOps Lab 01 - Churn API")


# ---------------------------------------------------------------------------
# Schéma d'entrée (Pydantic)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, le=200)
    num_complaints: int = Field(..., ge=0, le=50)
    avg_session_minutes: float = Field(..., ge=0.0, le=500.0)
    plan_type: str
    region: str
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Cache de modèle en mémoire
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {"name": None, "model": None}


# ---------------------------------------------------------------------------
# Fonctions utilitaires (chargement modèle, logging)
# ---------------------------------------------------------------------------

def get_current_model_name() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    return f"{MODEL_NAME}@{ALIAS} (v{mv.version})"


def load_model_if_needed() -> tuple[str, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    cache_key = MODEL_URI

    if _model_cache["name"] == cache_key and _model_cache["model"] is not None:
        return cache_key, _model_cache["model"]

    model = mlflow.sklearn.load_model(MODEL_URI)

    _model_cache["name"] = cache_key
    _model_cache["model"] = model
    return cache_key, model


def log_prediction(payload: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Endpoints FastAPI
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    try:
        model_name = get_current_model_name()
        return {"status": "ok", "current_model": model_name}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/startup")
def startup() -> dict[str, Any]:
    try:
        model_name = get_current_model_name()
        return {"status": "ok", "current_model": model_name}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/ready")
def ready() -> dict[str, Any]:
    try:
        model_name = get_current_model_name()
        return {"status": "ready", "current_model": model_name}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        model_name, model = load_model_if_needed()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    features = {
        "tenure_months": req.tenure_months,
        "num_complaints": req.num_complaints,
        "avg_session_minutes": req.avg_session_minutes,
        "plan_type": req.plan_type.strip().lower(),
        "region": req.region.strip().upper(),
    }

    X_df = pd.DataFrame([features])

    start = time.perf_counter()
    try:
        proba = float(model.predict_proba(X_df)[0][1])
        pred = int(proba >= 0.5)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000.0

    out: dict[str, Any] = {
        "request_id": req.request_id,
        "model_version": model_name,
        "prediction": pred,
        "probability": round(proba, 6),
        "latency_ms": round(latency_ms, 3),
        "features": features,
        "ts": int(time.time()),
    }

    log_prediction(out)
    return out
