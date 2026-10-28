import { useState } from "react";
import type { FormEvent } from "react";
import type { components } from "@repurposemap/api-client";

import { apiClient } from "../api.js";

type EvidenceLookupResponse = components["schemas"]["EvidenceLookupResponse"];
type PubMedEvidence = components["schemas"]["PubMedEvidence"];
type ClinicalTrialEvidence = components["schemas"]["ClinicalTrialEvidence"];
type SafetyEvidence = components["schemas"]["SafetyEvidence"];

type LookupState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "done"; result: EvidenceLookupResponse };

const SOURCE_LABEL: Record<string, string> = {
  pubmed: "PubMed",
  clinical_trials: "ClinicalTrials.gov",
  openfda: "openFDA",
  dailymed: "DailyMed",
};

export function EvidencePanel() {
  const [drugName, setDrugName] = useState("");
  const [diseaseName, setDiseaseName] = useState("");
  const [state, setState] = useState<LookupState>({ phase: "idle" });

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const drug = drugName.trim();
    const disease = diseaseName.trim();
    if (drug.length === 0 || disease.length === 0) {
      return;
    }
    setState({ phase: "loading" });
    try {
      const result = await apiClient.lookupEvidence({ drug_name: drug, disease_name: disease, max_results: 5 });
      setState({ phase: "done", result });
    } catch {
      setState({ phase: "error", message: "Could not reach the evidence API." });
    }
  }

  const disabled = state.phase === "loading";

  return (
    <section className="evidence-panel" aria-label="Evidence lookup">
      <h2>Evidence lookup</h2>
      <p className="hint">
        Live PubMed, ClinicalTrials.gov, openFDA and DailyMed records for a drug/disease pair. A record's presence is not
        proof of efficacy or safety -- see each item's own source.
      </p>

      <form className="query-form evidence-form" onSubmit={handleSubmit}>
        <div className="evidence-inputs">
          <div>
            <label htmlFor="evidence-drug-input">Drug</label>
            <input
              id="evidence-drug-input"
              type="text"
              value={drugName}
              onChange={(event) => setDrugName(event.target.value)}
              placeholder="e.g. sildenafil"
              disabled={disabled}
            />
          </div>
          <div>
            <label htmlFor="evidence-disease-input">Disease / condition</label>
            <input
              id="evidence-disease-input"
              type="text"
              value={diseaseName}
              onChange={(event) => setDiseaseName(event.target.value)}
              placeholder="e.g. pulmonary arterial hypertension"
              disabled={disabled}
            />
          </div>
        </div>
        <button type="submit" disabled={disabled || drugName.trim().length === 0 || diseaseName.trim().length === 0}>
          Look up evidence
        </button>
      </form>

      {state.phase === "loading" && <p className="hint">Querying PubMed, ClinicalTrials.gov, openFDA and DailyMed...</p>}
      {state.phase === "error" && <p className="error">{state.message}</p>}
      {state.phase === "done" && <EvidenceResult result={state.result} />}
    </section>
  );
}

function EvidenceResult({ result }: { result: EvidenceLookupResponse }) {
  const pubmed = result.bundle.pubmed ?? [];
  const clinicalTrials = result.bundle.clinical_trials ?? [];
  const safety = result.bundle.safety ?? [];

  return (
    <div className="evidence-result">
      <div className="source-status-row">
        {result.source_status.map((status) => (
          <span
            key={status.source}
            className={`source-status-chip ${status.succeeded ? "ok" : "unavailable"}`}
            title={status.error ?? undefined}
          >
            {SOURCE_LABEL[status.source] ?? status.source}: {status.succeeded ? "ok" : "failed"}
          </span>
        ))}
      </div>

      <EvidenceSection title="Literature (PubMed)">
        {pubmed.length === 0 ? (
          <p className="hint">No matching papers.</p>
        ) : (
          <ul>
            {pubmed.map((record) => (
              <PubMedItem key={record.pmid} record={record} />
            ))}
          </ul>
        )}
      </EvidenceSection>

      <EvidenceSection title="Clinical trials">
        {clinicalTrials.length === 0 ? (
          <p className="hint">No matching trials.</p>
        ) : (
          <ul>
            {clinicalTrials.map((record) => (
              <ClinicalTrialItem key={record.nct_id} record={record} />
            ))}
          </ul>
        )}
      </EvidenceSection>

      <EvidenceSection title="Safety (openFDA / DailyMed)">
        {safety.length === 0 ? (
          <p className="hint">No matching labels.</p>
        ) : (
          <ul>
            {safety.map((record) => (
              <SafetyItem key={`${record.source}-${record.source_id}`} record={record} />
            ))}
          </ul>
        )}
      </EvidenceSection>

      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}

function EvidenceSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="candidate-section">
      <h4>{title}</h4>
      {children}
    </section>
  );
}

function PubMedItem({ record }: { record: PubMedEvidence }) {
  return (
    <li>
      {record.url ? (
        <a href={record.url} target="_blank" rel="noreferrer">
          {record.title}
        </a>
      ) : (
        record.title
      )}
      {record.journal ? ` -- ${record.journal}` : ""}
      {record.publication_date ? ` (${record.publication_date})` : ""}
      {record.strength !== "unknown" ? <span className="score-chip">{record.strength}</span> : null}
    </li>
  );
}

function ClinicalTrialItem({ record }: { record: ClinicalTrialEvidence }) {
  return (
    <li>
      {record.url ? (
        <a href={record.url} target="_blank" rel="noreferrer">
          {record.title}
        </a>
      ) : (
        record.title
      )}
      {record.status ? ` -- ${record.status}` : ""}
      {record.phase ? ` (${record.phase})` : ""}
      {record.strength !== "unknown" ? <span className="score-chip">{record.strength}</span> : null}
    </li>
  );
}

function SafetyItem({ record }: { record: SafetyEvidence }) {
  return (
    <li>
      <span className="safety-flag">{SOURCE_LABEL[record.source] ?? record.source}:</span>{" "}
      {record.url ? (
        <a href={record.url} target="_blank" rel="noreferrer">
          {record.description}
        </a>
      ) : (
        record.description
      )}
    </li>
  );
}
