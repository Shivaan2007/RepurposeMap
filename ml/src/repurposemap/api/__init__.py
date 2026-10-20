"""The FastAPI boundary between the Python research engine and the future
TypeScript research workspace:

    Python research engine -> FastAPI + Pydantic -> OpenAPI
        -> generated TypeScript client -> future React research workspace

Today this package serves exactly two real endpoints (``GET /health``,
``GET /api/v1/system/info``) and defines the future candidate-ranking
contracts (``repurposemap.api.schemas``) so the OpenAPI schema -- and the
TypeScript types generated from it -- are stable from day one. No ranking
endpoint is wired to a real model yet; see ``system_info.ModelStatus``.

See the root ``API.md`` for how to run this locally, export the OpenAPI
schema, and regenerate the TypeScript client.
"""

from repurposemap.api.app import app, create_app

__all__ = ["app", "create_app"]
