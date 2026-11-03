import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { apiClient } from "../api.js";

/**
 * Compare 2-4 candidate drugs for one disease, side by side, using only
 * information already available today. There is no trained ranking model
 * yet (POST /api/v1/rank stays a stub), so "model score" and
 * "explanation-path count" are honestly reported as not available rather
 * than invented -- this view never computes a composite score or names a
 * "best" candidate. It is a researcher comparison aid, not a
 * recommendation.
 */

const MIN_CANDIDATES = 2;
const MAX_CANDIDATES = 4;

type ColumnState =
  | { phase: "pending" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | {
      phase: "done";
      pubmedCount: number;
      trialCount: number;
      safetyCount: number;
      sourceFailures: string[];
    };

interface ComparisonColumn {
  drugName: string;
  state: ColumnState;
}

interface CandidateComparisonPanelProps {
  /** Disease name from the main query above -- pre-fills this panel's own
   * field on each new search, but stays independently editable afterward. */
  initialDiseaseName?: string;
}

export function CandidateComparisonPanel({ initialDiseaseName }: CandidateComparisonPanelProps = {}) {
  const [diseaseName, setDiseaseName] = useState("");
  const [drugInputs, setDrugInputs] = useState<string[]>(["", ""]);
  const [columns, setColumns] = useState<ComparisonColumn[] | null>(null);

  useEffect(() => {
    if (initialDiseaseName) setDiseaseName(initialDiseaseName);
  }, [initialDiseaseName]);

  function updateDrugInput(index: number, value: string) {
    setDrugInputs((prev) => prev.map((existing, i) => (i === index ? value : existing)));
  }

  function addDrugInput() {
    if (drugInputs.length >= MAX_CANDIDATES) return;
    setDrugInputs((prev) => [...prev, ""]);
  }

  function removeDrugInput(index: number) {
    if (drugInputs.length <= MIN_CANDIDATES) return;
    setDrugInputs((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleCompare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const disease = diseaseName.trim();
    const drugs = drugInputs.map((d) => d.trim()).filter((d) => d.length > 0);
    if (disease.length === 0 || drugs.length < MIN_CANDIDATES) return;

    const initial: ComparisonColumn[] = drugs.map((drugName) => ({ drugName, state: { phase: "loading" } }));
    setColumns(initial);

    const results = await Promise.all(
      drugs.map(async (drugName): Promise<ComparisonColumn> => {
        try {
          const result = await apiClient.lookupEvidence({ drug_name: drugName, disease_name: disease, max_results: 5 });
          const sourceFailures = result.source_status.filter((s) => !s.succeeded).map((s) => s.source);
          return {
            drugName,
            state: {
              phase: "done",
              pubmedCount: result.bundle.pubmed?.length ?? 0,
              trialCount: result.bundle.clinical_trials?.length ?? 0,
              safetyCount: result.bundle.safety?.length ?? 0,
              sourceFailures,
            },
          };
        } catch {
          return { drugName, state: { phase: "error", message: "Could not reach the evidence API." } };
        }
      }),
    );
    setColumns(results);
  }

  const canAddMore = drugInputs.length < MAX_CANDIDATES;
  const canCompare = diseaseName.trim().length > 0 && drugInputs.filter((d) => d.trim().length > 0).length >= MIN_CANDIDATES;

  return (
    <section className="evidence-panel" aria-label="Candidate comparison">
      <h2>Candidate comparison</h2>
      <p className="hint">
        Compare 2-4 candidate drugs for one disease side by side, using only evidence-lookup counts available today.
        There is no trained ranking model yet, so no model score or explanation-path count is shown -- this never
        picks a "best" candidate.
      </p>

      <form className="query-form evidence-form" onSubmit={handleCompare}>
        <div className="evidence-inputs">
          <div>
            <label htmlFor="comparison-disease-input">Disease / condition</label>
            <input
              id="comparison-disease-input"
              type="text"
              value={diseaseName}
              onChange={(event) => setDiseaseName(event.target.value)}
              placeholder="e.g. pulmonary arterial hypertension"
            />
          </div>
        </div>

        {drugInputs.map((value, index) => (
          <div key={index} className="evidence-inputs comparison-drug-row">
            <div>
              <label htmlFor={`comparison-drug-input-${index}`}>Candidate drug {index + 1}</label>
              <input
                id={`comparison-drug-input-${index}`}
                type="text"
                value={value}
                onChange={(event) => updateDrugInput(index, event.target.value)}
                placeholder="e.g. sildenafil"
              />
            </div>
            {drugInputs.length > MIN_CANDIDATES && (
              <button type="button" className="notebook-remove" onClick={() => removeDrugInput(index)}>
                Remove
              </button>
            )}
          </div>
        ))}

        <div className="comparison-form-actions">
          <button type="button" onClick={addDrugInput} disabled={!canAddMore}>
            Add another candidate
          </button>
          <button type="submit" disabled={!canCompare}>
            Compare
          </button>
        </div>
      </form>

      {columns && <ComparisonTable columns={columns} />}
    </section>
  );
}

function ComparisonTable({ columns }: { columns: ComparisonColumn[] }) {
  return (
    <table className="comparison-table">
      <thead>
        <tr>
          <th>Candidate</th>
          {columns.map((column) => (
            <th key={column.drugName}>{column.drugName}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Model score</td>
          {columns.map((column) => (
            <td key={column.drugName} className="hint">
              not available (no trained model yet)
            </td>
          ))}
        </tr>
        <tr>
          <td>Explanation paths</td>
          {columns.map((column) => (
            <td key={column.drugName} className="hint">
              not available (no trained model yet)
            </td>
          ))}
        </tr>
        <tr>
          <td>PubMed records</td>
          {columns.map((column) => (
            <td key={column.drugName}>{renderCount(column.state, "pubmedCount")}</td>
          ))}
        </tr>
        <tr>
          <td>Clinical trials</td>
          {columns.map((column) => (
            <td key={column.drugName}>{renderCount(column.state, "trialCount")}</td>
          ))}
        </tr>
        <tr>
          <td>Safety records</td>
          {columns.map((column) => (
            <td key={column.drugName}>{renderCount(column.state, "safetyCount")}</td>
          ))}
        </tr>
        <tr>
          <td>Lookup status</td>
          {columns.map((column) => (
            <td key={column.drugName}>{renderStatus(column.state)}</td>
          ))}
        </tr>
      </tbody>
    </table>
  );
}

function renderCount(state: ColumnState, key: "pubmedCount" | "trialCount" | "safetyCount"): React.ReactNode {
  if (state.phase === "loading") return "loading...";
  if (state.phase === "error") return "unavailable";
  if (state.phase === "pending") return "-";
  return state[key];
}

function renderStatus(state: ColumnState): React.ReactNode {
  if (state.phase === "loading") return "loading...";
  if (state.phase === "pending") return "-";
  if (state.phase === "error") return <span className="error">{state.message}</span>;
  if (state.sourceFailures.length === 0) return "all sources answered";
  return <span className="hint">partial: {state.sourceFailures.join(", ")} unavailable</span>;
}
