import { useEffect, useState } from "react";
import type { components } from "@repurposemap/api-client";

import { apiClient } from "./api.js";
import { CandidateList } from "./components/CandidateList.js";
import { DiseaseBiologyPanel } from "./components/DiseaseBiologyPanel.js";
import { EvidencePanel } from "./components/EvidencePanel.js";
import { QueryForm } from "./components/QueryForm.js";
import { StatusPanel } from "./components/StatusPanel.js";

type SystemInfoResponse = components["schemas"]["SystemInfoResponse"];
type RankingResponse = components["schemas"]["RankingResponse"];

type QueryState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "done"; result: RankingResponse };

export function App() {
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [systemInfoError, setSystemInfoError] = useState<string | null>(null);
  const [queryState, setQueryState] = useState<QueryState>({ phase: "idle" });

  useEffect(() => {
    apiClient
      .systemInfo()
      .then(setSystemInfo)
      .catch(() => setSystemInfoError("API unreachable"));
  }, []);

  async function handleSubmit(diseaseName: string) {
    setQueryState({ phase: "loading" });
    try {
      const result = await apiClient.rankCandidates({ disease_name: diseaseName, top_k: 10 });
      setQueryState({ phase: "done", result });
    } catch {
      setQueryState({ phase: "error", message: "Could not reach the ranking API." });
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>RepurposeMap</h1>
        <p className="tagline">Explainable drug-repurposing research workspace</p>
      </header>

      <StatusPanel systemInfo={systemInfo} error={systemInfoError} />

      <QueryForm onSubmit={handleSubmit} disabled={queryState.phase === "loading"} />

      <section className="results" aria-live="polite">
        {queryState.phase === "idle" && (
          <p className="hint">Enter a disease or condition above, then generate and inspect drug-repurposing hypotheses.</p>
        )}
        {queryState.phase === "loading" && <p className="hint">Querying the ranking API...</p>}
        {queryState.phase === "error" && <p className="error">{queryState.message}</p>}
        {queryState.phase === "done" && <CandidateList result={queryState.result} />}
      </section>

      <DiseaseBiologyPanel />

      <EvidencePanel />

      <footer className="page-footer">
        <p>Research hypothesis ranking. Not a treatment recommendation. Every output requires expert review.</p>
      </footer>
    </div>
  );
}
