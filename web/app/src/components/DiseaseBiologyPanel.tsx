import { useState } from "react";
import type { FormEvent } from "react";
import type { components } from "@repurposemap/api-client";

import { RepurposeMapApiError } from "@repurposemap/api-client";

import { apiClient } from "../api.js";

type DiseaseBiologyResponse = components["schemas"]["DiseaseBiologyResponse"];
type TargetAssociation = components["schemas"]["TargetAssociation"];

type LookupState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "done"; result: DiseaseBiologyResponse };

export function DiseaseBiologyPanel() {
  const [diseaseName, setDiseaseName] = useState("");
  const [state, setState] = useState<LookupState>({ phase: "idle" });

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const disease = diseaseName.trim();
    if (disease.length === 0) {
      return;
    }
    setState({ phase: "loading" });
    try {
      const result = await apiClient.lookupDiseaseBiology({ disease_name: disease, max_targets: 10 });
      setState({ phase: "done", result });
    } catch (error) {
      if (error instanceof RepurposeMapApiError && error.status === 404) {
        setState({ phase: "error", message: "Open Targets found no matching disease." });
      } else {
        setState({ phase: "error", message: "Could not reach the disease-biology API." });
      }
    }
  }

  const disabled = state.phase === "loading";

  return (
    <section className="evidence-panel" aria-label="Disease biology">
      <h2>Disease biology</h2>
      <p className="hint">
        What goes wrong in this disease? Genes most strongly associated with it, ranked by genetic evidence, via Open
        Targets. An association is a research signal, not a causal claim.
      </p>

      <form className="query-form evidence-form" onSubmit={handleSubmit}>
        <div className="evidence-inputs">
          <div>
            <label htmlFor="biology-disease-input">Disease / condition</label>
            <input
              id="biology-disease-input"
              type="text"
              value={diseaseName}
              onChange={(event) => setDiseaseName(event.target.value)}
              placeholder="e.g. pulmonary arterial hypertension"
              disabled={disabled}
            />
          </div>
        </div>
        <button type="submit" disabled={disabled || diseaseName.trim().length === 0}>
          Look up disease biology
        </button>
      </form>

      {state.phase === "loading" && <p className="hint">Querying Open Targets...</p>}
      {state.phase === "error" && <p className="error">{state.message}</p>}
      {state.phase === "done" && <DiseaseBiologyResult result={state.result} />}
    </section>
  );
}

function DiseaseBiologyResult({ result }: { result: DiseaseBiologyResponse }) {
  const { module, resolved_from: resolvedFrom } = result;
  const targets = module.targets ?? [];

  return (
    <div className="evidence-result">
      {resolvedFrom && resolvedFrom.length > 0 && (
        <p className="hint">
          Resolved to <strong>{module.disease_name}</strong> ({module.disease_id})
          {resolvedFrom.length > 1 ? ` -- ${resolvedFrom.length} other candidates were considered` : ""}
        </p>
      )}

      {module.disease_description && <p className="hint">{module.disease_description}</p>}

      <section className="candidate-section">
        <h4>
          Associated genes ({targets.length} of {module.total_associated_targets})
        </h4>
        {targets.length === 0 ? (
          <p className="hint">No associated targets.</p>
        ) : (
          <ul>
            {targets.map((target) => (
              <TargetItem key={target.ensembl_id} target={target} />
            ))}
          </ul>
        )}
      </section>

      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}

function TargetItem({ target }: { target: TargetAssociation }) {
  const pathways = target.pathways ?? [];
  return (
    <li>
      <strong>{target.approved_symbol}</strong> -- {target.approved_name}
      <span className="score-chip">overall: {target.overall_score.toFixed(3)}</span>
      {target.genetic_association_score !== null && target.genetic_association_score !== undefined ? (
        <span className="score-chip">genetic: {target.genetic_association_score.toFixed(3)}</span>
      ) : null}
      {pathways.length > 0 ? (
        <div className="hint">pathways: {pathways.map((p) => p.pathway).join(", ")}</div>
      ) : null}
    </li>
  );
}
