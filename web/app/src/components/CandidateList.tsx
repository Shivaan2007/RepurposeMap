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
          <strong>Ranked candidates are not available for this disease in the current deployed dataset.</strong>
        </p>
        <p className="hint">
          TransE final training and its one locked-test evaluation are complete, and an R-GCN validation search is in
          progress -- but no R-GCN final model has been selected or trained yet. Separately, no precomputed ranking
          bundle has been exported for this disease yet (see <code>GET /api/v1/rank/diseases</code> for the current
          list). This page will show ranked candidates once either exists -- it never substitutes a mock or random
          ranking.
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
      {result.serving_mode === "precomputed" && (
        <p className="hint ranking-provenance">
          <strong>Precomputed ranking</strong> -- method: {result.ranking_method ?? "unknown"}; candidate universe:{" "}
          {result.candidate_universe ?? "unspecified"}; generated: {result.generated_at ?? "unknown"}. Not a live
          model inference result.
        </p>
      )}
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
        <h4>Explanation paths</h4>
        {explanationPaths.length === 0 ? (
          <p className="hint">No explanation path recorded.</p>
        ) : (
          <>
            {explanationPaths.slice(0, 4).map((path, index) => {
              const steps = (path.steps ?? []).slice(0, 4);
              return (
                <ol key={index} className="explanation-path">
                  <li>
                    {steps[0] ? `${steps[0].source_name} (${steps[0].source_type})` : null}
                  </li>
                  {steps.map((step, stepIndex) => (
                    <li key={stepIndex}>
                      --[{step.display_relation ?? step.relation}]--&gt; {step.target_name} ({step.target_type})
                    </li>
                  ))}
                  {path.path_score != null ? (
                    <li className="hint">path quality score (heuristic, not a probability): {path.path_score.toFixed(3)}</li>
                  ) : null}
                </ol>
              );
            })}
            <p className="hint">
              Graph connectivity is not evidence of efficacy. A path shows that these entities and relations exist in
              PrimeKG, not that the drug treats the disease.
            </p>
          </>
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
