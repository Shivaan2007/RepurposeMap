# Demo/portfolio materials (current V1 state)

Supporting material for showing RepurposeMap to someone else -- a
screenshot checklist, two narration scripts at different lengths, and a
GitHub project description. Companion to `docs/DEMO_SCRIPT.md` (the full
click-through demo walkthrough) and `docs/PORTFOLIO.md` (resume bullets,
tech stack, research/engineering contributions).

State as of 2026-11-28: the frozen R-GCN has been evaluated once on the locked test (see
[`docs/FINAL_RESULTS.md`](FINAL_RESULTS.md)), 5 precomputed R-GCN rankings are served, and the app is live at
https://repurposemap.vercel.app (API: https://repurposemap-api.onrender.com). Same ground rule as before: state only what
is true, and **never** describe the rankings as live model inference, as evidence that a drug works, or as a
validated benchmark. Update this file, `docs/DEMO_SCRIPT.md` and `docs/PORTFOLIO.md` together if anything changes.

## Screenshots

Captured on 2026-11-28 from the live deployment in a real browser (1280 px wide, light mode), with real data in every
field. Nothing is mocked or edited.

| File | What it shows |
|---|---|
| [`01-status-and-query.png`](screenshots/01-status-and-query.png) | System status ("API: Connected") and the list of diseases that have a deployed ranking. "Model: Unavailable" means live inference is unavailable; the rankings are precomputed. |
| [`02-ranked-candidates.png`](screenshots/02-ranked-candidates.png) | Lead image: a real ranked candidate list for "breast neoplasm", labelled "Precomputed ranking ... Not a live model inference result". Scores only; no explanation paths. |
| [`03-not-available.png`](screenshots/03-not-available.png) | The honest empty state for a disease with no ranking. |
| [`04-disease-biology.png`](screenshots/04-disease-biology.png) | Disease biology for breast cancer: real Open Targets genes (top of the panel). |
| [`05-target-scorecard.png`](screenshots/05-target-scorecard.png) | Target scorecard for ERBB2 from Open Targets. |
| [`06-evidence-lookup.png`](screenshots/06-evidence-lookup.png) | Evidence lookup for doxorubicin and breast cancer: PubMed, ClinicalTrials.gov, openFDA and DailyMed all "ok" (top of the panel; the safety section below it is much longer). |

Not captured: the Markdown brief download, the candidate comparison panel and the hypothesis notebook.

## 60-90 second demo script (elevator pitch)

For a context where you have one breath to make the point -- a portfolio
video intro, a hallway conversation, the first 90 seconds of a longer demo.

> RepurposeMap is a research tool for finding existing drugs that might
> treat a disease with no approved treatment -- drug repurposing. It
> searches a 4-million-edge biomedical knowledge graph, and for any
> candidate it finds, it pulls in live evidence: has anyone studied this
> combination, is there a clinical trial, is there a safety warning. The
> part I actually care about most is that it never pretends to know
> something it doesn't -- [open the status panel] this tells you honestly
> whether a real ranking model is available right now, instead of showing
> a plausible-looking fake number. That's a deliberate design choice, not
> a missing feature: a wrong confident number in a drug-discovery tool is
> worse than an honest "not yet." [open a ranked disease, e.g. breast neoplasm]
> This is a real ranking from a relational graph neural network I trained and
> evaluated exactly once on a locked set of 100 held-out diseases -- it beats my
> TransE baseline (Hits@10 0.062 vs 0.016), but it is still weak in absolute terms
> and it is precomputed for five diseases, not live. [one real query -- disease
> biology or evidence lookup] That part is live, real data, right now.

(Everything in the brackets is a stage direction; the numbers are from
`docs/FINAL_RESULTS.md`. Do not add a claim that it finds new treatments.)

Re-check the numbers against `docs/FINAL_RESULTS.md` before using this.

## 2-minute technical walkthrough

For an audience that will ask "how does the ranking actually work" --
engineers, technical interviewers. Assumes `docs/DEMO_SCRIPT.md`'s full
click-through as the backing material; this is the narration to go with
steps 1-3 and 9 of it, compressed.

> The core idea: PrimeKG gives you a huge, typed biomedical graph -- drugs,
> diseases, genes, pathways, millions of edges, each edge typed by relation.
> I built three ways to rank candidate drugs for a disease on top of it.
> The simplest is a path-quality heuristic: find paths between drug and
> disease through the graph, score them by relation type and penalize hub
> nodes and promiscuous drugs that connect to everything. That's explainable
> by construction -- you get the actual path as the explanation -- but it
> only ranks the bounded candidate set it can reach by search, not the full
> drug universe. Second is popularity, a dumb baseline: rank by raw graph
> degree, full 7,957-drug universe, no learning at all -- it exists so any
> learned model has something real to beat. Third is TransE, a translational
> knowledge-graph embedding model -- I ran a validation search, trained the
> winner, and evaluated it exactly once against a locked, held-out test set
> of 100 diseases that no model selection ever touched. It didn't beat the
> path baseline on the locked test, which is itself a real result: a pure
> embedding model isn't picking up enough of PrimeKG's relational structure
> on its own. That's what motivated the next model, R-GCN -- a relational
> graph neural network that conditions directly on edge types instead of
> compressing everything into one embedding geometry. I picked its configuration
> on validation only, trained it once, and evaluated it once on the same locked
> test: Hits@10 0.062 and MRR 0.033, against 0.016 and 0.010 for TransE. By the
> rule I fixed in advance -- both metrics have to be higher -- that counts as an
> improvement over TransE. It is not better than the path baseline on Hits@10,
> and those two rank different-sized candidate sets, so I don't claim a ranking
> between them; and with 498 pairs I can't give a significance interval, because
> I never saved TransE's per-disease results. The model's output is exported
> through a serving layer so the API and UI read precomputed files, not a live
> model -- five diseases today.

## GitHub project description

Short (for the repo's "About" field, ~350 characters):

> Explainable drug-repurposing research tool over a 4M-edge biomedical
> knowledge graph (PrimeKG). Path-quality, popularity, TransE and R-GCN rankers
> evaluated on a leakage-safe, disease-held-out, single-locked-test protocol.
> FastAPI + React, live PubMed/ClinicalTrials.gov/openFDA/DailyMed/Open Targets
> evidence. Research hypotheses only, not treatment recommendations.

One-line topics/tags: `drug-repurposing`, `knowledge-graph`,
`graph-neural-networks`, `fastapi`, `react`, `bioinformatics`,
`explainable-ai`.
