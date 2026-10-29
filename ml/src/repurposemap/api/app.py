"""The FastAPI application factory.

Run locally with:

    uvicorn repurposemap.api.app:app --reload

See ml/src/repurposemap/api/__init__.py and the root API.md for the full
picture (health/system-info today, the future ranking contract, OpenAPI
export, and the generated TypeScript client).
"""

from __future__ import annotations

from fastapi import FastAPI

from repurposemap.api.disease_biology import router as disease_biology_router
from repurposemap.api.evidence import router as evidence_router
from repurposemap.api.health import router as health_router
from repurposemap.api.ranking import router as ranking_router
from repurposemap.api.system_info import router as system_info_router
from repurposemap.api.version import API_VERSION, APPLICATION_NAME


def create_app() -> FastAPI:
    app = FastAPI(
        title=f"{APPLICATION_NAME} API",
        version=API_VERSION,
        description=(
            "Explainable drug-repurposing research API. Every ranking output is a "
            "research hypothesis for expert review, never a treatment recommendation."
        ),
    )
    app.include_router(health_router)
    app.include_router(system_info_router)
    app.include_router(ranking_router)
    app.include_router(evidence_router)
    app.include_router(disease_biology_router)
    return app


app = create_app()
