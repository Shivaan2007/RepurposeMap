import { useState } from "react";
import type { FormEvent } from "react";

interface QueryFormProps {
  onSubmit: (diseaseName: string) => void;
  disabled: boolean;
}

export function QueryForm({ onSubmit, disabled }: QueryFormProps) {
  const [diseaseName, setDiseaseName] = useState("");

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
    </form>
  );
}
