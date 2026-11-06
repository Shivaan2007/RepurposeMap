import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { components } from "@repurposemap/api-client";

import { apiClient } from "../api.js";

type SupportedDiseaseSummary = components["schemas"]["SupportedDiseaseSummary"];

interface QueryFormProps {
  onSubmit: (diseaseName: string) => void;
  disabled: boolean;
}

export function QueryForm({ onSubmit, disabled }: QueryFormProps) {
  const [diseaseName, setDiseaseName] = useState("");
  const [supportedDiseases, setSupportedDiseases] = useState<SupportedDiseaseSummary[] | null>(null);

  useEffect(() => {
    apiClient
      .listSupportedDiseases()
      .then((response) => setSupportedDiseases(response.diseases ?? []))
      .catch(() => setSupportedDiseases([]));
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = diseaseName.trim();
    if (trimmed.length === 0) {
      return;
    }
    onSubmit(trimmed);
  }

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <label htmlFor="disease-input">Disease / condition</label>
      <div className="query-row">
        <input
          id="disease-input"
          type="text"
          value={diseaseName}
          onChange={(event) => setDiseaseName(event.target.value)}
          placeholder="e.g. idiopathic pulmonary fibrosis"
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || diseaseName.trim().length === 0}>
          Analyze hypothesis
        </button>
      </div>
      {supportedDiseases !== null && (
        <p className="hint">
          {supportedDiseases.length === 0
            ? "No disease has a precomputed ranking bundle deployed yet -- any query will honestly report ranked candidates as not available."
            : `Diseases with a precomputed ranking available today: ${supportedDiseases.map((d) => d.disease_name).join(", ")}.`}
        </p>
      )}
    </form>
  );
}
