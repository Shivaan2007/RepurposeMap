import { RepurposeMapClient } from "@repurposemap/api-client";

// VITE_API_BASE_URL points the generated client at a deployed backend on
// its own origin (see DEMO.md). With no env var set, baseUrl falls back to
// "" (same-origin) -- in local dev that's the Vite dev server, which
// proxies /api and /health to the FastAPI service (see vite.config.ts); a
// same-origin production build is expected to be served behind the same
// reverse proxy as the API.
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

export const apiClient = new RepurposeMapClient({ baseUrl: apiBaseUrl });
