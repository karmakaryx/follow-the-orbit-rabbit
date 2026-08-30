import os
import tempfile
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import LSTMAutoencoder

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "models/")
CACHE_TTL_MINUTES = float(os.getenv("CACHE_TTL_MINUTES", "15"))

app = FastAPI(title="FTOR Collision Anomaly Scoring")

# 차후 React UI를 위한 CORS 설정. 지금은 전체 허용
# TODO: 도메인이 정해지면 CORS_ALLOWED_ORIGINS 환경변수로 좁힐 것
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_cors_origins] if _cors_origins != "*" else ["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# 모델 1개 + scaler + processed 데이터 캐시를 프로세스 전역에 들고 있음
_state = {"model": None, "scaler": None, "df": None, "df_loaded_at": None}


class ScoreResponse(BaseModel):
    norad_cat_id: int
    object_name: str | None
    anomaly_score: float
    num_snapshots_used: int
    latest_epoch: str


def _s3_client():
    return boto3.client("s3")


def _find_latest_checkpoint_key(bucket: str, prefix: str) -> str:
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        candidates.extend(o for o in page.get("Contents", []) if o["Key"].endswith(".ckpt"))
    if not candidates:
        raise RuntimeError(f"❌ No checkpoint found in S3: s3://{bucket}/{prefix}")
    best = max(candidates, key=lambda o: o["LastModified"])
    return best["Key"]


def _load_model_and_scaler() -> tuple[LSTMAutoencoder, dict]:
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


def _list_recent_processed_keys(bucket: str, lookback_hours: float) -> list[str]:
    now = datetime.now(timezone.utc)
    days_back = int(lookback_hours // 24) + 2  # 여유 있게 이틀 더
    candidate_dates = {(now - timedelta(days=d)).date() for d in range(days_back + 1)}
    prefixes = [
        f"processed/year={d.year:04d}/month={d.month:02d}/day={d.day:02d}/" for d in candidate_dates
    ]
    s3 = _s3_client()
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(o["Key"] for o in page.get("Contents", []))
    return keys


def _refresh_processed_cache():
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.


@app.on_event("startup")
def on_startup():
    _state["model"], _state["scaler"] = _load_model_and_scaler()
    _refresh_processed_cache()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/score/{norad_cat_id}", response_model=ScoreResponse)
def score(norad_cat_id: int):
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.
