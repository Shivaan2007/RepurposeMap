/**
 * A minimal, typed fetch wrapper over the RepurposeMap FastAPI service.
 * No frontend framework, no extra runtime dependencies -- just `fetch`
 * plus the types in ./types.ts.
 */

import type { components } from "./types.js";

type DiseaseQuery = components["schemas"]["DiseaseQuery"];
type HealthResponse = components["schemas"]["HealthResponse"];
type RankingResponse = components["schemas"]["RankingResponse"];
type SystemInfoResponse = components["schemas"]["SystemInfoResponse"];

export class RepurposeMapApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown,
  ) {
    super(message);
    this.name = "RepurposeMapApiError";
  }
}

export interface RepurposeMapClientOptions {
  /** e.g. "http://localhost:8000" */
  baseUrl: string;
  fetchFn?: typeof fetch;
}

export class RepurposeMapClient {
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;

  constructor(options: RepurposeMapClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.fetchFn = options.fetchFn ?? fetch;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchFn(`${this.baseUrl}${path}`, {
      headers: { "content-type": "application/json", ...init?.headers },
      ...init,
    });
    const body = (await response.json().catch(() => null)) as unknown;
    if (!response.ok) {
      throw new RepurposeMapApiError(`${init?.method ?? "GET"} ${path} failed with ${response.status}`, response.status, body);
    }
    return body as T;
  }

  health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  systemInfo(): Promise<SystemInfoResponse> {
    return this.request<SystemInfoResponse>("/api/v1/system/info");
  }

  /**
   * Always returns `model_status: "not_available"` with an empty
   * candidate list until a real model is wired into the server --
   * this client never fabricates a ranking locally either.
   */
  rankCandidates(query: DiseaseQuery): Promise<RankingResponse> {
    return this.request<RankingResponse>("/api/v1/rank", {
      method: "POST",
      body: JSON.stringify(query),
    });
  }
}
