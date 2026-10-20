/**
 * Types mirroring ../../openapi.json's current contract.
 *
 * NOTE ON PROVENANCE: these were meant to be generated mechanically by
 * `npm run generate` (openapi-typescript), but this file was written by
 * hand instead -- the environment this was built in could not reach
 * registry.npmjs.org to install openapi-typescript. The shapes below were
 * read directly from openapi.json, not guessed. Once you have normal
 * network access, run:
 *
 *     npm install
 *     npm run generate
 *
 * which will overwrite this file with a mechanically generated one from
 * the real openapi-typescript tool. Until then, if you change a Pydantic
 * schema in ml/src/repurposemap/api/schemas.py, re-export openapi.json
 * (see the root API.md) and update this file to match by hand.
 */

// --- Live today: /health, /api/v1/system/info ---

export interface HealthResponse {
  service: string;
  status: "ok";
  api_version: string;
}

export interface ModelStatus {
  available: boolean;
  /** e.g. "path-quality", "transe" -- null until a model is configured */
  model_type: string | null;
  ready_for_inference: boolean;
}

export interface Provenance {
  source: string;
  retrieved_date: string | null;
  version: string | null;
}

export interface SystemInfoResponse {
  application: string;
  api_version: string;
  model: ModelStatus;
  data_provenance: Provenance | null;
}

// --- Future candidate-ranking contract (POST /api/v1/rank, stubbed) ---

export interface DiseaseQuery {
  disease_name?: string | null;
  disease_id?: string | null;
  /** 1-100, default 10 */
  top_k?: number;
}

export interface SafetyFlag {
  code: string;
  severity: "info" | "caution" | "exclude";
  description: string;
}

export interface EvidenceSummary {
  relation: string;
  display_relation: string;
  source_node_type: string;
  target_node_type: string;
}

export interface ExplanationStep {
  source_id: string;
  source_name: string;
  relation: string;
  target_id: string;
  target_name: string;
}

export interface ExplanationPath {
  steps: ExplanationStep[];
  path_score: number | null;
}

export interface CandidateScore {
  value: number;
  /** e.g. "path_quality", "popularity", "transe" */
  method: string;
  rank: number | null;
}

export interface CandidateDrug {
  drug_id: string;
  drug_name: string;
  scores: CandidateScore[];
  evidence: EvidenceSummary[];
  explanation_paths: ExplanationPath[];
  safety_flags: SafetyFlag[];
}

export type ModelAvailability = "not_available" | "ready";

export interface RankingResponse {
  disease_query: DiseaseQuery;
  candidates: CandidateDrug[];
  model_status: ModelAvailability;
  provenance: Provenance | null;
  disclaimer: string;
}
