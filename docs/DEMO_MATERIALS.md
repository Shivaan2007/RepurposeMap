# Demo/portfolio materials (current V1 state)

Supporting material for showing RepurposeMap to someone else -- a
screenshot checklist, two narration scripts at different lengths, and a
GitHub project description. Companion to `docs/DEMO_SCRIPT.md` (the full
click-through demo walkthrough) and `docs/PORTFOLIO.md` (resume bullets,
tech stack, research/engineering contributions).

Same ground rule as those two files: **nothing here states an R-GCN
locked-test result, a "final model," or a live-deployment claim**, because
none of those are true yet. Update this file, `docs/DEMO_SCRIPT.md`, and
`docs/PORTFOLIO.md` together once they are.

## Screenshot checklist

Capture these from a local run (`DEMO.md`'s setup) at a normal laptop
browser width, light mode, with real (not placeholder) data in every field:

- [ ] Status panel showing `GET /api/v1/system/info`'s honest
      `model_available: false` state -- this is the "never fakes a ready
      model" screenshot, worth keeping even after R-GCN ships, relabeled.
- [ ] Main query box with the list of diseases that have a deployed ranking
      bundle shown above it (`GET /api/v1/rank/diseases`) -- today this list
      is empty; re-capture once `data/serving/` has real bundles.
- [ ] A disease search result showing the honest "not available for this
      disease" explanation (not an empty table).
- [ ] Disease biology panel with a real Open Targets gene/pathway result.
- [ ] Target scorecard panel for one of those genes.
- [ ] Evidence lookup panel for a real drug/disease pair, showing which of
      PubMed/ClinicalTrials.gov/openFDA/DailyMed succeeded.
- [ ] The "Download brief (Markdown)" button mid-click or the downloaded
      file's content, to show the export is real.
- [ ] Candidate comparison panel with 2-3 drugs side by side, showing
      "Model score: not available" stated explicitly rather than blank.
- [ ] Hypothesis notebook with one tracked candidate and a status/note.
- [ ] Once R-GCN ships: the same main-query screenshot, now showing a real
      ranked candidate list with scores and explanation paths -- this
      replaces the "not available" screenshot above as the lead image, which
      moves to an appendix/before-after comparison instead of being deleted.

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
> worse than an honest "not yet." [one real query -- disease biology or
> evidence lookup] This part is live, real data, right now. The ranking
> model itself -- [name current status from
> `V1_RELEASE_CHECKLIST.md`/`README.md`'s "Project status"] -- is the piece
> I'm finishing next.

Swap the bracketed status line for whatever's true when you give this --
don't let it go stale.

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
> compressing everything into one embedding geometry -- [current status].
> Whatever wins gets exported through a serving layer I built specifically
> so that dropping in a real trained model's output is a data step, not a
> code change -- the API and UI are already wired end-to-end against that
> contract, they're just honestly reporting "not available" until a real
> bundle exists.

## GitHub project description

Short (for the repo's "About" field, ~350 characters):

> Explainable drug-repurposing research tool over a 4M-edge biomedical
> knowledge graph (PrimeKG). Path-quality, popularity, TransE and (in
> progress) R-GCN rankers evaluated on a leakage-safe, disease-held-out,
> single-locked-test protocol. FastAPI + React, live PubMed/ClinicalTrials.gov/
> openFDA/DailyMed/Open Targets evidence. Research hypotheses only, not
> treatment recommendations.

One-line topics/tags: `drug-repurposing`, `knowledge-graph`,
`graph-neural-networks`, `fastapi`, `react`, `bioinformatics`,
`explainable-ai`.
