"""CORS origin configuration.

Allowed origins come from the ``REPURPOSEMAP_CORS_ORIGINS`` environment
variable: a comma-separated list of origins, e.g.

    REPURPOSEMAP_CORS_ORIGINS=https://app.example.com,https://staging.example.com

When unset, a development default is used -- the Vite dev server's own
origin (see ``web/app/vite.config.ts``) -- so local development works with
no configuration. ``*`` is never combined with credentials: this API does
not use cookies/credentialed requests, so ``allow_credentials`` stays
``False`` regardless of configured origins.
"""

from __future__ import annotations

import os

DEFAULT_DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

CORS_ORIGINS_ENV_VAR = "REPURPOSEMAP_CORS_ORIGINS"


def get_allowed_origins() -> list[str]:
    """Parse ``REPURPOSEMAP_CORS_ORIGINS``, falling back to dev defaults."""
    raw = os.environ.get(CORS_ORIGINS_ENV_VAR)
    if raw is None or raw.strip() == "":
        return list(DEFAULT_DEV_ORIGINS)
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins
