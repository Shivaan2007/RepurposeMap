/**
 * A minimal, typed fetch wrapper over the RepurposeMap FastAPI service.
 * No frontend framework, no extra runtime dependencies -- just `fetch`
 * plus the types in ./types.ts.
 */

import type { components } from "./types.js";

type DiseaseBiologyQuery = components["schemas"]["DiseaseBiologyQuery"];
type DiseaseBiologyResponse = components["schemas"]["DiseaseBiologyResponse"];
type DiseaseQuery = components["schemas"]["DiseaseQuery"];
type EvidenceLookupResponse = components["schemas"]["EvidenceLookupResponse"];
type EvidenceQuery = components["schemas"]["EvidenceQuery"];
type HealthResponse = components["schemas"]["HealthResponse"];
type RankingResponse = components["schemas"]["RankingResponse"];
type SystemInfoResponse = components["schemas"]["SystemInfoResponse"];
type TargetScorecardQuery = components["schemas"]["TargetScorecardQuery"];
type TargetScorecardResponse = components["schemas"]["TargetScorecardResponse"];

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

  /**
   * A real, synchronous lookup across PubMed, ClinicalTrials.gov, openFDA
   * and DailyMed for one drug/disease pair. Always returns 200 -- a
   * source that failed is reported in `source_status` with its error,
   * not folded into an empty-but-successful result. See the server-side
   * router's docstring for why this is synchronous for now.
   */
  lookupEvidence(query: EvidenceQuery): Promise<EvidenceLookupResponse> {
    return this.request<EvidenceLookupResponse>("/api/v1/evidence", {
      method: "POST",
      body: JSON.stringify(query),
    });
  }

  /**
   * A real, synchronous lookup of the genes most strongly associated
   * with a disease (ranked by genetic evidence), via Open Targets.
   * Unlike `lookupEvidence`, a real failure throws `RepurposeMapApiError`
   * (404 for an unresolvable name/unknown ID, 502 for an Open Targets
   * failure) rather than returning 200 with an empty module -- there is
   * only one source here, so there is no partial result to report.
   */
  lookupDiseaseBiology(query: DiseaseBiologyQuery): Promise<DiseaseBiologyResponse> {
    return this.request<DiseaseBiologyResponse>("/api/v1/disease-biology", {
      method: "POST",
      body: JSON.stringify(query),
    });
  }

  /**
   * A real, synchronous lookup of one target's tractability, Open
   * Targets' own prioritisation factors, and its highest-expressing
   * tissues, with no disease context. Same error-shape convention as
   * `lookupDiseaseBiology`: throws on a real failure rather than
   * returning 200 with an empty scorecard.
   */
  lookupTargetScorecard(query: TargetScorecardQuery): Promise<TargetScorecardResponse> {
    return this.request<TargetScorecardResponse>("/api/v1/target-scorecard", {
      method: "POST",
      body: JSON.stringify(query),
    });
  }
}
