import type { components } from "@repurposemap/api-client";

type SystemInfoResponse = components["schemas"]["SystemInfoResponse"];

interface StatusPanelProps {
  systemInfo: SystemInfoResponse | null;
  error: string | null;
}

export function StatusPanel({ systemInfo, error }: StatusPanelProps) {
  const apiConnected = systemInfo !== null;
  const modelAvailable = systemInfo?.model.available ?? false;

  return (
    <section className="status-panel" aria-label="System status">
      <h2>System status</h2>
      <dl>
        <div className="status-row">
          <dt>API</dt>
          <dd className={apiConnected ? "ok" : "unavailable"}>{error ?? (apiConnected ? "Connected" : "Checking...")}</dd>
        </div>
        <div className="status-row">
          <dt>Model</dt>
          <dd className={modelAvailable ? "ok" : "unavailable"}>{modelAvailable ? "Available" : "Unavailable"}</dd>
        </div>
        <div className="status-row">
          <dt>Dataset</dt>
          <dd>{systemInfo?.data_provenance?.source ?? "PrimeKG"}</dd>
        </div>
        <div className="status-row">
          <dt>Current model stage</dt>
          <dd>{systemInfo?.model_provenance?.most_recent_completed_stage ?? "Not yet available"}</dd>
        </div>
      </dl>
    </section>
  );
}
