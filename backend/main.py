"""FastAPI application exposing the existing churn workflow to the React client."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from churn import data as D

from .schemas import RiskBandsRequest, TrainRequest
from .service import ApiError, DashboardService, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB

ALLOWED_UPLOADS = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"}
SESSION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")


def create_app(service: Optional[DashboardService] = None) -> FastAPI:
    dashboard = service or DashboardService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await asyncio.to_thread(dashboard.initialize)
        yield

    app = FastAPI(
        title="Churn Intelligence API",
        version="1.0.0",
        description="FastAPI adapter over the existing churn prediction and retention logic.",
        lifespan=lifespan,
    )
    origins = [value.strip() for value in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if value.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Session-ID"],
        expose_headers=["X-Session-ID"],
    )

    @app.middleware("http")
    async def attach_session_id(request: Request, call_next):
        session_id = request.headers.get("X-Session-ID", "")
        if not SESSION_RE.fullmatch(session_id):
            session_id = str(uuid.uuid4())
        request.state.session_id = session_id
        response = await call_next(request)
        response.headers["X-Session-ID"] = session_id
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    def sid(request: Request) -> str:
        return request.state.session_id

    async def read_upload(file: UploadFile) -> tuple[str, str, object]:
        filename = (file.filename or "upload.csv").replace("\\", "/").split("/")[-1]
        suffix = os.path.splitext(filename)[1].lower()
        if suffix and suffix not in ALLOWED_UPLOADS:
            raise ApiError(415, "Supported uploads: CSV, TSV, TXT, XLSX, XLSM, XLS.")
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise ApiError(413, f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.")
        try:
            frame = D.load_table(BytesIO(contents), filename=filename)
        except D.DataLoadError as exc:
            raise ApiError(422, str(exc)) from exc
        return filename, hashlib.sha256(contents).hexdigest(), frame

    @app.get("/api/health")
    def health():
        dashboard.initialize()
        return {
            "status": "ok",
            "model_loaded": dashboard.pkg is not None,
            "training_data_loaded": dashboard.train_df is not None,
            "errors": dashboard.startup_errors,
        }

    @app.get("/api/bootstrap")
    def bootstrap(request: Request):
        return dashboard.bootstrap(sid(request))

    @app.get("/api/overview")
    def overview(request: Request):
        return dashboard.overview(sid(request))

    @app.get("/api/customers")
    def customers(
        request: Request,
        query: str = "",
        risk_levels: Optional[list[str]] = Query(None),
        probability_min: float = Query(0.0, ge=0.0, le=1.0),
        probability_max: float = Query(1.0, ge=0.0, le=1.0),
        segment: Optional[str] = None,
        segment_values: Optional[list[str]] = Query(None),
        sort_by: str = "probability",
        descending: bool = True,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return dashboard.customers(
            sid(request), query=query, risk_levels=risk_levels,
            probability_min=probability_min, probability_max=probability_max,
            segment=segment, segment_values=segment_values, sort_by=sort_by,
            descending=descending, limit=limit, offset=offset,
        )

    @app.get("/api/customers/{customer_id}")
    def customer_detail(customer_id: str, request: Request):
        return dashboard.customer_detail(sid(request), customer_id)

    @app.get("/api/retention")
    def retention(
        request: Request,
        risk_levels: Optional[list[str]] = Query(None),
        rule_ids: Optional[list[str]] = Query(None),
        segment: Optional[str] = None,
        segment_values: Optional[list[str]] = Query(None),
        sort_by: str = "probability",
        limit: int = Query(100, ge=1, le=1000),
    ):
        return dashboard.retention(
            sid(request), risk_levels=risk_levels, rule_ids=rule_ids,
            segment=segment, segment_values=segment_values,
            sort_by=sort_by, limit=limit,
        )

    @app.get("/api/model-insights")
    def model_insights(request: Request, fraction: float = Query(0.1, ge=0.01, le=0.5)):
        return dashboard.model_insights(sid(request), fraction)

    @app.get("/api/workspace/status")
    def workspace_status(request: Request):
        return dashboard.workspace_status(sid(request))

    @app.post("/api/workspace/train-upload")
    async def train_upload(request: Request, file: UploadFile = File(...)):
        filename, digest, frame = await read_upload(file)
        return dashboard.store_train_upload(sid(request), filename, digest, frame)

    @app.post("/api/workspace/train")
    def train(request: Request, payload: TrainRequest):
        return dashboard.train(sid(request), payload)

    @app.post("/api/workspace/score-upload")
    async def score_upload(request: Request, file: UploadFile = File(...)):
        filename, digest, frame = await read_upload(file)
        return dashboard.score_upload(sid(request), filename, digest, frame)

    @app.get("/api/workspace/scores.csv")
    def score_download(request: Request):
        content = dashboard.score_csv(sid(request))
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="churn_scores.csv"'},
        )

    @app.post("/api/workspace/bands")
    def set_bands(request: Request, payload: RiskBandsRequest):
        return dashboard.set_bands(sid(request), payload.low_max, payload.high_min)

    @app.post("/api/workspace/reset")
    def reset(request: Request):
        return dashboard.reset(sid(request))

    app.state.dashboard = dashboard
    return app


app = create_app()
