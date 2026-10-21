import type { components } from "@repurposemap/api-client";

type RankingResponse = components["schemas"]["RankingResponse"];
type CandidateDrug = components["schemas"]["CandidateDrug"];

interface CandidateListProps {
  result: RankingResponse;
}

export function CandidateList({ result }: CandidateListProps) {
  if (result.model_status === "not_available") {
    return (
      <div className="candidates-empty">
        <p>
          <strong>No ranking model is available yet.</strong>
        </p>
        <p className="hint">
          Final TransE training has not completed and no eligible checkpoint is configured. This page will show ranked
          candidates once a model is wired in -- it never substitutes a mock or random ranking.
        </p>
      </div>
    );
  }

  const candidates = result.candidates ?? [];
  if (candidates.length === 0) {
    return <p className="hint">No candidates were found for this query.</p>;
  }

  return (
    <div className="candidate-table">
      {candidates.map((candidate) => (
        <CandidateRow key={candidate.drug_id} candidate={candidate} />
      ))}
    </div>
  );
}

function CandidateRow({ candidate }: { candidate: CandidateDrug }) {
  const scores = candidate.scores ?? [];
  const evidence = candidate.evidence ?? [];
  const explanationPaths = candidate.explanation_paths ?? [];
  const safetyFlags = candidate.safety_flags ?? [];

  return (
    <article className="candidate-row">
      <header className="candidate-header">
        <h3>{candidate.drug_name}</h3>
        <div className="candidate-scores">
          {scores.map((score) => (
            <span key={score.method} className="score-chip">
              {score.method}: {score.value.toFixed(3)}
            </span>
          ))}
        </div>
      </header>

      <section className="candidate-section">
        <h4>Evidence</h4>
        {evidence.length === 0 ? (
          <p className="hint">No graph evidence recorded.</p>
        ) : (
          <ul>
            {evidence.map((item, index) => (
              <li key={index}>
                {item.source_node_type} --[{item.display_relation}]--&gt; {item.target_node_type}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="candidate-section">
        <h4>Explanation path</h4>
        {explanationPaths.length === 0 ? (
          <p className="hint">No explanation path recorded.</p>
        ) : (
          explanationPaths.map((path, index) => (
            <ol key={index} className="explanation-path">
              {(path.steps ?? []).map((step, stepIndex) => (
                <li key={stepIndex}>
                  {step.source_name} --[{step.relation}]--&gt; {step.target_name}
                </li>
              ))}
            </ol>
          ))
        )}
      </section>

      <section className="candidate-section">
        <h4>Safety</h4>
        {safetyFlags.length === 0 ? (
          <p className="hint">No safety flags recorded.</p>
        ) : (
          <ul>
            {safetyFlags.map((flag) => (
              <li key={flag.code} className={`safety-flag severity-${flag.severity}`}>
                {flag.description}
              </li>
            ))}
          </ul>
        )}
      </section>
    </article>
  );
}
