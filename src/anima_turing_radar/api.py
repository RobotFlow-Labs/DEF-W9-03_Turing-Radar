"""FastAPI service for Turing-Radar inference."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .constants import NUM_PDW_FEATURES
from .model import build_deinterleaver
from .preprocess import PDWPreprocessor


class PredictRequest(BaseModel):
    pulses: list[list[float]] = Field(min_length=1)
    algorithm: str | None = None


class PredictResponse(BaseModel):
    labels: list[int]
    n_clusters: int


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    pre = PDWPreprocessor.from_config(cfg.preprocess)
    base_model = build_deinterleaver(cfg.clustering)

    app = FastAPI(title="ANIMA Turing-Radar API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str | bool]:
        return {"ready": True, "algorithm": cfg.clustering.algorithm}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        x = np.asarray(req.pulses, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != NUM_PDW_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected pulses shape [N,{NUM_PDW_FEATURES}], got {list(x.shape)}",
            )

        algo = req.algorithm or cfg.clustering.algorithm
        model = base_model
        if algo != cfg.clustering.algorithm:
            model = build_deinterleaver(replace(cfg.clustering, algorithm=algo))

        x_proc = pre.fit_transform(x)
        labels = model.predict(x_proc)
        n_clusters = len(set(labels.tolist()) - {-1})
        return PredictResponse(labels=labels.tolist(), n_clusters=n_clusters)

    return app


def run() -> None:  # pragma: no cover
    cfg = load_config()
    uvicorn.run(create_app(), host=cfg.service.host, port=cfg.service.port)


if __name__ == "__main__":  # pragma: no cover
    run()
