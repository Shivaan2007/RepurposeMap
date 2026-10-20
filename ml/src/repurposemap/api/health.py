"""``GET /health`` -- liveness only. No graph load, no model check."""

from __future__ import annotations

from fastapi import APIRouter

from repurposemap.api.schemas import HealthResponse
from repurposemap.api.version import API_VERSION

SERVICE_NAME = "repurposemap-api"

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(service=SERVICE_NAME, status="ok", api_version=API_VERSION)
