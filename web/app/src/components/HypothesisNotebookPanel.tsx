import { useEffect, useState } from "react";
import type { FormEvent } from "react";

/**
 * A local, per-browser researcher notebook: track candidate drug/disease
 * hypotheses with a status and free-text notes while reviewing them. This
 * is deliberately NOT a backend feature -- there is no Postgres table or
 * API endpoint behind it. Entries live only in this browser's
 * localStorage, never reach the server, and are lost if the browser's
 * site data is cleared or a different device/browser is used. If a real
 * multi-device/shared notebook is wanted later, that needs a database and
 * an API, which PLAN.md does not yet call for ahead of R-GCN's result.
 */

const STORAGE_KEY = "repurposemap.hypothesis-notebook.v1";

type HypothesisStatus = "idea" | "reviewing" | "planned_experiment" | "dropped";

const STATUS_LABEL: Record<HypothesisStatus, string> = {
  idea: "Idea",
  reviewing: "Reviewing",
  planned_experiment: "Planned experiment",
  dropped: "Dropped",
};

const STATUS_ORDER: HypothesisStatus[] = ["idea", "reviewing", "planned_experiment", "dropped"];

interface HypothesisEntry {
  id: string;
  drugName: string;
  diseaseName: string;
  status: HypothesisStatus;
  notes: string;
  createdAt: string;
}

function loadEntries(): HypothesisEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveEntries(entries: HypothesisEntry[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Best-effort only -- a private window or blocked site data means this
    // session's notebook just won't persist across a reload.
  }
}

interface HypothesisNotebookPanelProps {
  /** Disease name from the main query above -- pre-fills the "add
   * hypothesis" field on each new search, but stays independently
   * editable afterward. */
  initialDiseaseName?: string;
}

export function HypothesisNotebookPanel({ initialDiseaseName }: HypothesisNotebookPanelProps = {}) {
  const [entries, setEntries] = useState<HypothesisEntry[]>([]);
  const [drugName, setDrugName] = useState("");
  const [diseaseName, setDiseaseName] = useState("");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setEntries(loadEntries());
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (initialDiseaseName) setDiseaseName(initialDiseaseName);
  }, [initialDiseaseName]);

  useEffect(() => {
    if (loaded) saveEntries(entries);
  }, [entries, loaded]);

  function handleAdd(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const drug = drugName.trim();
    const disease = diseaseName.trim();
    if (drug.length === 0 || disease.length === 0) return;
    const entry: HypothesisEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      drugName: drug,
      diseaseName: disease,
      status: "idea",
      notes: "",
      createdAt: new Date().toISOString(),
    };
    setEntries((prev) => [entry, ...prev]);
    setDrugName("");
    setDiseaseName("");
  }

  function updateEntry(id: string, patch: Partial<HypothesisEntry>) {
    setEntries((prev) => prev.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)));
  }

  function removeEntry(id: string) {
    setEntries((prev) => prev.filter((entry) => entry.id !== id));
  }

  return (
    <section className="evidence-panel" aria-label="Hypothesis notebook">
      <h2>Hypothesis notebook</h2>
      <p className="hint">
        Track candidate drug/disease hypotheses while you review them -- idea, reviewing, planned experiment, or
        dropped, plus your own notes. Stored only in this browser (localStorage), not on the server -- it is lost if
        this browser's site data is cleared, and is not shared with any other device or collaborator.
      </p>

      <form className="query-form evidence-form" onSubmit={handleAdd}>
        <div className="evidence-inputs">
          <div>
            <label htmlFor="notebook-drug-input">Drug</label>
            <input
              id="notebook-drug-input"
              type="text"
              value={drugName}
              onChange={(event) => setDrugName(event.target.value)}
              placeholder="e.g. sildenafil"
            />
          </div>
          <div>
            <label htmlFor="notebook-disease-input">Disease / condition</label>
            <input
              id="notebook-disease-input"
              type="text"
              value={diseaseName}
              onChange={(event) => setDiseaseName(event.target.value)}
              placeholder="e.g. pulmonary arterial hypertension"
            />
          </div>
        </div>
        <button type="submit" disabled={drugName.trim().length === 0 || diseaseName.trim().length === 0}>
          Add hypothesis
        </button>
      </form>

      {entries.length === 0 ? (
        <p className="hint">No hypotheses tracked yet.</p>
      ) : (
        <ul className="notebook-list">
          {entries.map((entry) => (
            <li key={entry.id} className="notebook-entry">
              <div className="notebook-entry-header">
                <strong>
                  {entry.drugName} &rarr; {entry.diseaseName}
                </strong>
                <button type="button" className="notebook-remove" onClick={() => removeEntry(entry.id)}>
                  Remove
                </button>
              </div>
              <div className="evidence-inputs">
                <div>
                  <label htmlFor={`notebook-status-${entry.id}`}>Status</label>
                  <select
                    id={`notebook-status-${entry.id}`}
                    value={entry.status}
                    onChange={(event) => updateEntry(entry.id, { status: event.target.value as HypothesisStatus })}
                  >
                    {STATUS_ORDER.map((status) => (
                      <option key={status} value={status}>
                        {STATUS_LABEL[status]}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <label htmlFor={`notebook-notes-${entry.id}`}>Notes</label>
              <textarea
                id={`notebook-notes-${entry.id}`}
                value={entry.notes}
                onChange={(event) => updateEntry(entry.id, { notes: event.target.value })}
                rows={2}
                placeholder="Your own research notes -- never shown anywhere else."
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
