# Release text, ready to paste once the real numbers exist

Nothing here is a result. Every `{{FILL: ...}}` marker names the file that
supplies its value. `python scripts/post_checkpoint.py placeholders` lists any
marker or `TBD` still left in the release-facing files, and `qa` fails while any
remain. Delete this file's unused variants when you paste.

Where the numbers come from after the single locked-test run:
`ml/experiments/rgcn_final_locked_test_<sha8>.json` (metrics, provenance) and, if
the significance step could run, `ml/results/rgcn_vs_transe_significance_<sha8>.json`.

---

## A. README: "Final R-GCN result (locked test)"

Paste after the existing "Final TransE result (locked test)" section.

> ### Final R-GCN result (locked test)
>
> The frozen sampled R-GCN (`rgcn-wide`, selected on validation only, trained once on the final training graph for 3 epochs) was evaluated against the same 100 locked test diseases / 498 held-out pairs **exactly once**. Provenance and metrics: [`ml/experiments/rgcn_final_locked_test_{{FILL: sha8}}.json`](ml/experiments/rgcn_final_locked_test_{{FILL: sha8}}.json). Protocol, fixed before any R-GCN result existed: [`ml/RGCN_EVALUATION_PROTOCOL.md`](ml/RGCN_EVALUATION_PROTOCOL.md).
>
> | | Hits@1 | Hits@3 | Hits@10 | MRR |
> |---|---|---|---|---|
> | Path baseline (bounded candidates) | 0.01004 | 0.04016 | 0.08635 | 0.03329 |
> | Popularity baseline (full universe, *approximate*) | 0.0 | ~0.004 | ~0.014 | ~0.0062 |
> | Final TransE (full universe) | 0.0020 | 0.0040 | 0.0161 | 0.0100 |
> | **Final R-GCN (full universe)** | {{FILL: hits_at_1}} | {{FILL: hits_at_3}} | {{FILL: hits_at_10}} | {{FILL: mrr}} |
>
> **Candidate-universe caveat.** The path baseline ranks a bounded, per-disease discovered candidate set (0 to 500 candidates); popularity, TransE and R-GCN rank the full 7,957-drug universe. The rows share the same held-out pairs and metric formulas but are not the same ranking task, so no "N times better" claim between the path baseline and the others should be made from this table.
>
> **Interpretation.** Pick exactly one of I-1, I-2, I-3 by the frozen decision rule (an improvement over TransE requires **both** MRR above 0.0100 **and** Hits@10 above 0.0161), then delete the others.

**I-1 (both higher).** R-GCN scored above TransE on both MRR ({{FILL: mrr}} vs 0.0100) and Hits@10 ({{FILL: hits_at_10}} vs 0.0161), so by the pre-registered rule it counts as an improvement over TransE on this split. {{FILL: one sentence on the significance interval, from variant S-1 or S-2 below}} Absolute performance is still {{FILL: describe honestly, e.g. weak: the true drug lands in the top 10 for about N in 100 held-out pairs}}. This does not establish clinical efficacy for any candidate.

**I-2 (neither higher).** R-GCN did not improve on TransE under the pre-registered rule (MRR {{FILL: mrr}} vs 0.0100; Hits@10 {{FILL: hits_at_10}} vs 0.0161). It is reported as it came out; nothing was retuned or rerun. {{FILL: one honest sentence on what that suggests about this split and model, without speculation presented as a finding}} This does not establish clinical efficacy for any candidate.

**I-3 (one higher).** R-GCN was higher than TransE on {{FILL: MRR or Hits@10}} but not on {{FILL: the other}}, so by the pre-registered rule (both required) it does **not** count as an improvement. Numbers: MRR {{FILL: mrr}} vs 0.0100; Hits@10 {{FILL: hits_at_10}} vs 0.0161. This does not establish clinical efficacy for any candidate.

**S-1 (significance computed).** A disease-level paired bootstrap over the 100 locked-test diseases ({{FILL: n_resamples}} resamples, seed 42) gives a 95% interval for the R-GCN minus TransE difference of {{FILL: MRR interval}} on MRR and {{FILL: Hits@10 interval}} on Hits@10; the MRR interval {{FILL: excludes / includes}} zero and the Hits@10 interval {{FILL: excludes / includes}} zero.

**S-2 (significance not computed).** The paired bootstrap in `docs/FINAL_RESULTS.md` §3 needs TransE's per-disease outcomes, which were not saved when TransE's locked test ran (only aggregate metrics were), and rerunning that evaluation was not authorised. So no significance interval is reported and the comparison with TransE rests on the point estimates alone, on 498 pairs from 100 diseases.

---

## B. README: "Live demo" block

> ## Live demo
>
> - Frontend: {{FILL: Vercel URL}}
> - API: {{FILL: Render URL}} (`/health`, `/api/v1/system/info`, `/openapi.json`)
> - The free API tier sleeps when idle, so the first request after a pause can take a while.
> - Candidate rankings are **precomputed** for {{FILL: N}} diseases and served from files, not computed live (`serving_mode: "precomputed"`). Any other disease honestly returns `model_status: "not_available"`.
>
> ![Main query]({{FILL: path to the real screenshot, from docs/DEMO_MATERIALS.md}})

---

## C. README: status and roadmap lines after closure

Replace the "In progress / not yet complete" block with:

> **Completed (V1)**
> - Sampled R-GCN validation search, final training and the single locked-test evaluation: result in [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md).
> - Precomputed ranking serving for {{FILL: N}} diseases, masked case studies ({{FILL: N}}) in `case_studies/`, deployment on Render and Vercel.
>
> **Not done / out of scope for V1**
> - Live model inference, the triage shortlist, the results page in the frontend, the time-travel test, the TxGNN comparison, expert review.

---

## D. `docs/PORTFOLIO.md`: replace "Results placeholder"

> ## Results
> The one locked-test result for the frozen R-GCN, next to the path, popularity and TransE baselines, is in [`docs/FINAL_RESULTS.md`](FINAL_RESULTS.md). Headline: {{FILL: one sentence copied from FINAL_RESULTS.md §1}}. Caveats stated there: a small test set (100 diseases, 498 pairs), a path baseline that ranks a different candidate set, and no claim of clinical efficacy.

---

## E. Resume bullets (from PLAN.md; fill only with real values, drop any bullet whose number does not exist)

```
\resumeItem{Built an explainable drug repurposing tool on PrimeKG (8.1M edges), ranking existing drugs for diseases held out of training with a relational GNN trained in PyTorch; Hits@10 of {{FILL: R-GCN hits_at_10}} and MRR of {{FILL: mrr}} on 100 locked-test diseases, against {{FILL: TransE 0.0161 / path 0.0863}} for the baselines, evaluated once under a pre-registered protocol with reverse edges removed}
\resumeItem{Added typed evidence and safety lookups against PubMed, ClinicalTrials.gov, openFDA, DailyMed and Open Targets, a FastAPI + OpenAPI backend, and a React/TypeScript research workspace, with CI, mocked-network tests and a deployed demo ({{FILL: URL}})}
\resumeItem{Ran {{FILL: N}} masked case studies: removed a known indication from the model's message-passing graph at inference time and checked where it ranks ({{FILL: ranks}}); the weights were not retrained, so this is a weaker check than a retrain-and-replay}
```
Do not add the time-travel bullet, the expert-review bullet or any TxGNN claim: none was run. **Case-study limitation to state wherever case studies are cited:** `ml/scripts/run_case_study.py` masks the known edge when encoding, but the weights were trained with that edge, and PLAN.md's version (delete the edge, retrain, re-rank) was not run.

---

## F. One results sentence for the pitch and GitHub description

> Evaluated once on a locked test of 100 held-out diseases, the model reaches Hits@10 = {{FILL: hits_at_10}} (TransE {{FILL: or 0.0161}}); a research hypothesis generator, not a treatment recommendation.
