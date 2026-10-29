import { useState } from "react";
import type { FormEvent } from "react";
import type { components } from "@repurposemap/api-client";

import { RepurposeMapApiError } from "@repurposemap/api-client";

import { apiClient } from "../api.js";

type TargetScorecardResponse = components["schemas"]["TargetScorecardResponse"];

type LookupState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "done"; result: TargetScorecardResponse };

export function TargetScorecardPanel() {
  const [geneSymbol, setGeneSymbol] = useState("");
  const [state, setState] = useState<LookupState>({ phase: "idle" });

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const gene = geneSymbol.trim();
    if (gene.length === 0) {
      return;
    }
    setState({ phase: "loading" });
    try {
      const result = await apiClient.lookupTargetScorecard({ gene_symbol: gene });
      setState({ phase: "done", result });
    } catch (error) {
      if (error instanceof RepurposeMapApiError && error.status === 404) {
        setState({ phase: "error", message: "Open Targets found no matching gene." });
      } else {
        setState({ phase: "error", message: "Could not reach the target-scorecard API." });
      }
    }
  }

  const disabled = state.phase === "loading";

  return (
    <section className="evidence-panel" aria-label="Target scorecard">
      <h2>Target scorecard</h2>
      <p className="hint">
        Which proteins are worth hitting? Tractability, Open Targets' own target-prioritisation factors, and the
        tissues where a gene is most highly expressed -- no disease context.
      </p>

      <form className="query-form evidence-form" onSubmit={handleSubmit}>
        <div className="evidence-inputs">
          <div>
            <label htmlFor="scorecard-gene-input">Gene symbol</label>
            <input
              id="scorecard-gene-input"
              type="text"
              value={geneSymbol}
              onChange={(event) => setGeneSymbol(event.target.value)}
              placeholder="e.g. BMPR2"
              disabled={disabled}
            />
          </div>
        </div>
        <button type="submit" disabled={disabled || geneSymbol.trim().length === 0}>
          Look up target scorecard
        </button>
      </form>

      {state.phase === "loading" && <p className="hint">Querying Open Targets...</p>}
      {state.phase === "error" && <p className="error">{state.message}</p>}
      {state.phase === "done" && <TargetScorecardResult result={state.result} />}
    </section>
  );
}

function TargetScorecardResult({ result }: { result: TargetScorecardResponse }) {
  const { scorecard, resolved_from: resolvedFrom } = result;
  const tractability = scorecard.tractability ?? [];
  const tractableModalities = [...new Set(tractability.filter((t) => t.value).map((t) => t.modality))];
  const factors = Object.entries(scorecard.prioritisation_factors ?? {});
  const tissues = scorecard.top_tissue_expression ?? [];

  return (
    <div className="evidence-result">
      {resolvedFrom && resolvedFrom.length > 0 && (
        <p className="hint">
          Resolved to <strong>{scorecard.approved_symbol}</strong> ({scorecard.ensembl_id})
          {resolvedFrom.length > 1 ? ` -- ${resolvedFrom.length} other candidates were considered` : ""}
        </p>
      )}
      <p className="hint">{scorecard.approved_name}</p>

      <section className="candidate-section">
        <h4>Tractability</h4>
        {tractableModalities.length === 0 ? (
          <p className="hint">No tractability criteria were met.</p>
        ) : (
          <p className="hint">Tractable modalities: {tractableModalities.join(", ")}</p>
        )}
        {tractability.length > 0 && (
          <ul>
            {tractability
              .filter((t) => t.value)
              .map((t, index) => (
                <li key={`${t.modality}-${t.label}-${index}`}>
                  <span className="score-chip">{t.modality}</span> {t.label}
                </li>
              ))}
          </ul>
        )}
      </section>

      <section className="candidate-section">
        <h4>Prioritisation factors</h4>
        {factors.length === 0 ? (
          <p className="hint">No prioritisation factors reported.</p>
        ) : (
          <ul>
            {factors.map(([key, value]) => (
              <li key={key}>
                {key}: {value.toFixed(3)}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="candidate-section">
        <h4>Highest-expressing tissues</h4>
        {tissues.length === 0 ? (
          <p className="hint">No tissue expression data.</p>
        ) : (
          <ul>
            {tissues.map((t, index) => (
              <li key={`${t.tissue}-${index}`}>
                {t.tissue}: {t.median?.toFixed(2)} {t.unit}
              </li>
            ))}
          </ul>
        )}
      </section>

      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}
