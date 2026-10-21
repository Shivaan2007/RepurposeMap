import { RepurposeMapClient } from "@repurposemap/api-client";

// Same-origin: the Vite dev server proxies /api and /health to the
// FastAPI service (see vite.config.ts); a production build is expected
// to be served behind the same reverse proxy as the API.
export const apiClient = new RepurposeMapClient({ baseUrl: "" });
