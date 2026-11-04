import { useEffect, useState } from "react";
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

interface EvidencePanelProps {
  /** Disease name from the main query above -- pre-fills this panel's own
   * field on each new search, but stays independently editable afterward. */
  initialDiseaseName?: string;
}

export function EvidencePanel({ initialDiseaseName }: EvidencePanelProps = {}) {
  const [drugName, setDrugName] = useState("");
  const [diseaseName, setDiseaseName] = useState("");
  const [state, setState] = useState<LookupState>({ phase: "idle" });

  useEffect(() => {
    if (initialDiseaseName) setDiseaseName(initialDiseaseName);
  }, [initialDiseaseName]);

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
      {state.phase === "done" && <EvidenceResult result={state.result} drugName={drugName} diseaseName={diseaseName} />}
    </section>
  );
}

function buildEvidenceBriefMarkdown(
  result: EvidenceLookupResponse,
  drugName: string,
  diseaseName: string,
): string {
  const pubmed = result.bundle.pubmed ?? [];
  const clinicalTrials = result.bundle.clinical_trials ?? [];
  const safety = result.bundle.safety ?? [];
  const lines: string[] = [];

  lines.push(`# Evidence brief: ${drugName} for ${diseaseName}`);
  lines.push("");
  lines.push(`Generated ${new Date().toISOString()} by RepurposeMap. ${result.disclaimer}`);
  lines.push("");
  lines.push("## Source status");
  for (const status of result.source_status) {
    const label = SOURCE_LABEL[status.source] ?? status.source;
    lines.push(`- ${label}: ${status.succeeded ? "ok" : `failed (${status.error ?? "unknown error"})`}`);
  }

  lines.push("");
  lines.push(`## Literature (PubMed) -- ${pubmed.length}`);
  if (pubmed.length === 0) {
    lines.push("No matching papers.");
  } else {
    for (const record of pubmed) {
      const parts = [record.journal, record.publication_date, record.strength !== "unknown" ? record.strength : null].filter(
        Boolean,
      );
      const suffix = parts.length > 0 ? ` (${parts.join(", ")})` : "";
      const titleMarkdown = record.url ? `[${record.title}](${record.url})` : record.title;
      lines.push(`- ${titleMarkdown}${suffix}`);
    }
  }

  lines.push("");
  lines.push(`## Clinical trials -- ${clinicalTrials.length}`);
  if (clinicalTrials.length === 0) {
    lines.push("No matching trials.");
  } else {
    for (const record of clinicalTrials) {
      const parts = [record.nct_id, record.status, record.phase, record.strength !== "unknown" ? record.strength : null].filter(
        Boolean,
      );
      const suffix = parts.length > 0 ? ` (${parts.join(", ")})` : "";
      const titleMarkdown = record.url ? `[${record.title}](${record.url})` : record.title;
      lines.push(`- ${titleMarkdown}${suffix}`);
    }
  }

  lines.push("");
  lines.push(`## Safety (openFDA / DailyMed) -- ${safety.length}`);
  if (safety.length === 0) {
    lines.push("No matching labels.");
  } else {
    for (const record of safety) {
      const label = SOURCE_LABEL[record.source] ?? record.source;
      const warning = record.boxed_warning_present ? " **boxed warning**" : "";
      lines.push(`- ${label}: ${record.description}${warning}`);
      if (record.contraindication_text) {
        lines.push(`  - Contraindications: ${record.contraindication_text}`);
      }
    }
  }

  lines.push("");
  lines.push(
    "_Research hypothesis only, not a treatment recommendation. A record's presence is not proof of efficacy or safety -- see each item's own source._",
  );
  return lines.join("\n");
}

function downloadMarkdown(filename: string, markdown: string): void {
  const blob = new Blob([markdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function EvidenceResult({
  result,
  drugName,
  diseaseName,
}: {
  result: EvidenceLookupResponse;
  drugName: string;
  diseaseName: string;
}) {
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
        <button
          type="button"
          className="notebook-remove"
          onClick={() => {
            const slug = `${drugName}-${diseaseName}`.toLowerCase().replace(/[^a-z0-9]+/g, "-");
            downloadMarkdown(`evidence-brief-${slug}.md`, buildEvidenceBriefMarkdown(result, drugName, diseaseName));
          }}
        >
          Download brief (Markdown)
        </button>
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
      {record.nct_id ? ` (${record.nct_id})` : ""}
      {record.status ? ` -- ${record.status}` : ""}
      {record.phase ? ` (${record.phase})` : ""}
      {record.strength !== "unknown" ? <span className="score-chip">{record.strength}</span> : null}
      {record.sponsor || record.enrollment != null ? (
        <p className="hint">
          {record.sponsor ? `Sponsor: ${record.sponsor}` : ""}
          {record.sponsor && record.enrollment != null ? " -- " : ""}
          {record.enrollment != null ? `Enrollment: ${record.enrollment}` : ""}
        </p>
      ) : null}
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
      {record.boxed_warning_present ? <span className="score-chip safety-flag">boxed warning</span> : null}
      {record.contraindication_text ? <p className="hint">Contraindications: {record.contraindication_text}</p> : null}
    </li>
  );
}
