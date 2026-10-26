
##
RepurposeMap Build Guide (v2)
Updated 9 Oct 2026 · @Shivaan Bhalla
What changed in this version
This version fixes eight things in the 4 Oct guide so the results hold up when someone checks them. Everything else is the same plan.
#	Change	Where it lives now	Why
1	Time-travel test uses the PrimeKG build date as the cutoff	How we test it	PrimeKG is one snapshot and most edges have no date, so you can't truly use only older knowledge
2	Popularity baseline added	How we test it, KPIs	Famous drugs show up in new papers and trials anyway; beating random drugs isn't enough
3	TxGNN's disease splits used for the comparison	The model, How we test it	Makes "compare with TxGNN" a real like-for-like number
4	Case studies mask the known answer first	How we test it	Sildenafil for PAH is probably already in the graph, so replaying it proves nothing until you hide it
5	Reverse edges of test links removed at split time	The model	Otherwise the model sees the answer through the flipped relation
6	Signature reversal (step 4) demoted	Researcher workflow, Schedule	LINCS is big and awkward, and most rare diseases have no disease signature
7	Hand-labeled abstracts fixed at 150	KPIs, How we test it	The old guide said 100 in one place and 150 in another
8	Expert review and one focus disease area added	KPIs, Schedule, Showing it off	A domain expert looking at your top candidates adds more credibility than another feature
What it is
RepurposeMap suggests existing drugs that might help diseases with no treatment, explains each suggestion as a path through biology, and checks papers, trials and safety data so researchers get a short list worth testing. Version 1 lands in early January 2027 at about 10 hours a week.
Most rare diseases have no approved treatment, and making a new drug takes over a decade. A faster route is finding a new use for a drug that already exists, called drug repurposing. Sildenafil is the classic case: developed for chest pain, now also approved for a serious lung condition.
RepurposeMap works on a giant map of biology where drugs, genes, proteins, pathways and diseases are connected. You type a disease; an AI model finds existing drugs that connect to it, shows the route it followed, then checks papers, clinical trials and safety data so a researcher sees which ideas are new, plausible and not obviously unsafe.
What makes it yours: published models like TxGNN (Nature Medicine, 2024) already predict candidates. RepurposeMap adds the step researchers do by hand today: checking every suggestion against real evidence and safety, and turning the result into a short, shareable brief.
What it is not: a treatment recommendation. Every result is a research idea for an expert to judge.
Focus area (new): pick one disease area before December and go deep on it, for example rare neurological diseases, since the brain-penetration step already points that way. The search still works on any disease in the graph, but the case studies, the evidence briefs and the expert review all come from that one area.
How it works
The model does the first half. The three checks are what turn a raw ranking into something a researcher can act on.
1.	Search. The user types a disease and it is matched to a node in the graph.
2.	Rank. The model scores every drug in the graph for that disease.
3.	Explain. For each top drug, the 3 strongest paths from drug to disease are kept and drawn.
4.	Check the evidence. PubMed and ClinicalTrials.gov are searched and the idea is labeled Novel, Some early evidence, or Already tested.
5.	Check the safety. Label warnings and contraindications are pulled and compared with the disease.
6.	Check the mechanism. Where data exists, the drug's effect on gene activity is compared with the disease's (optional, see the researcher workflow).
7.	Triage. Model score, evidence and safety are combined into one ranked shortlist and a one-page brief.
Features
Core (Version 1)
Feature	What it does
Disease search	Type a disease name; matches it to the graph, including rare-disease synonyms
Ranked candidates	Every drug in the graph scored for that disease
Reasoning paths	The 3 strongest routes from drug to disease, drawn as a clickable graph
Evidence check	Searches PubMed and ClinicalTrials.gov; labels each idea Novel, Some early evidence, or Already tested
Safety check	Pulls label warnings and contraindications; flags conflicts with the disease
Triage shortlist	Combines model score, evidence and safety into one ranked list
Evidence brief	One-page summary to download and share
Results page	All the test results, published
Stretch (Version 2)
Feature	What it does
Drug view	Pick a drug and see every disease it might help
Watch list	Save a disease; get notified when new papers or trials appear for your top candidates
Paper reader	NLP reads each matched abstract and tags whether it supports, contradicts or just mentions the idea
Compare with TxGNN	Side-by-side with the published model's predictions, on the same disease splits
Deliberately left out
•	Patient-facing advice of any kind.
•	"This drug will work" language. The app shows reasons and evidence only.
•	Dosing information.
Researcher workflow
RepurposeMap follows the steps a drug discovery scientist takes before proposing a repurposing study. Each step is a tab in the app, and each one answers the question a reviewer would ask.
Step	The scientist's question	What RepurposeMap shows	Public data	Build
1. Disease biology	What goes wrong in this disease?	Disease module: genes, pathways and tissues linked to the disease, ranked by strength of genetic evidence	PrimeKG, Open Targets Platform	After core
2. Target prioritization	Which proteins are worth hitting?	Target scorecard: genetic evidence, tissue expression, whether a drug can bind it (tractability)	Open Targets, PrimeKG anatomy links	After core
3. Hypothesis generation	Which existing drugs hit those targets or that biology?	Ranked candidates from the graph model, with reasoning paths	PrimeKG + your GNN	First
4. Mechanism direction check	Does the drug push the biology the right way?	Signature reversal score: does the drug's effect on gene activity look like the opposite of the disease's? Flags drugs that would make things worse	Public LINCS drug signatures, published disease signatures	Optional (see note)
5. Evidence review	What's already known?	Papers sorted by evidence level: cell study, animal study, case report, clinical trial, with the key sentence highlighted	PubMed, ClinicalTrials.gov	First
6. Can it reach the tissue?	Will the drug get where it's needed?	Predicted brain penetration for neuro diseases, plus known tissue distribution	MoleculeNet BBBP, drug labels	Later
7. Safety profile	What could go wrong?	Label warnings, top reported side effects, predicted heart-rhythm risk (hERG), predicted toxicity	DailyMed, openFDA, TDC hERG, Tox21	First
8. Validation plan	How would I test this cheaply?	Suggested lab models: cell lines where the target is active, and existing mouse models of the disease	DepMap, mouse model databases	Later
9. Context	Who would care?	Disease prevalence, orphan drug designations already granted, patent status	Orphanet, FDA orphan designations, Orange Book	Later
Why step 4 moved down. LINCS is a big, awkward dataset, and most rare diseases have no published disease signature, so the check would only work for a small slice of diseases. Use it for the case studies and for any disease that has a published signature, and report how many diseases it covered as a count. Don't let it block the core build.
Scientist-style tools across every step
•	Hypothesis notebook: save a candidate, add notes, track its status (Idea, Reviewing, Planned experiment, Dropped, with reason).
•	Side-by-side scorecard: compare 2 to 4 candidates on every step at once.
•	Provenance on every claim: every number and sentence links to its source and data version, so nothing is unsupported.
•	Export: evidence brief as PDF, citations as BibTeX or RIS for reference managers, and the full table as CSV.
•	Case studies: three known repurposing successes (for example sildenafil for pulmonary arterial hypertension) replayed through the workflow with the known answer hidden first, to show it would have found them.
Scope. The researcher workflow adds about 80 hours on top of Version 1. Build steps 3, 5 and 7 first; they are the standout features. Steps 1, 2, 6, 8 and 9 come after, and step 4 only as described above.
KPIs
The numbers that go on the results page and the resume.
Area	KPI	What good looks like
Model	Hits@10, MRR, AUPRC on held-out treatments	Beats path and TransE baselines
Model	Hits@10 on diseases with no treatments seen	Clear gain over baselines
Model	Same metrics on TxGNN's disease splits	Reported next to TxGNN's published numbers
Real-world signal	Time-travel hit rate: share of top-50 ideas that later appeared in trials or papers, on pairs never studied before the cutoff	Well above both a random-drug baseline and a popularity baseline
Mechanism	Share of known treatments with a positive signature reversal score, plus the number of diseases it covered	High where it applies; coverage reported as a count
Evidence	Precision and recall of evidence-level labels against 150 hand-labeled abstracts	High precision first; it's what researchers trust
Safety	Share of candidates flagged by a known contraindication or boxed warning	Reported, not hidden
Case studies	Rank of the real drug in each replayed success story, after its known edge is masked and the model retrained	Top 10
Coverage	Rare diseases with at least one candidate passing every check	Reported as a count
Expert review	Of the top 10 candidates for the focus disease, how many a domain expert calls plausible enough to test	Reported as a count, with the expert's role named
Trust	Claims with a linked source	100%
Speed	Time to build an evidence brief: by hand vs with RepurposeMap	Measure it yourself on 3 diseases
Data sources
All free and public.
Source	What it gives you	Used for
PrimeKG (Harvard)	About 17,000 diseases and over 4 million relationships between drugs, genes, proteins, pathways, side effects and diseases; includes known indications and contraindications	The knowledge graph the model learns from
TxGNN code and splits	The disease splits and published results from the Nature Medicine paper	Like-for-like comparison
PubMed API (NIH)	Search over biomedical papers	Evidence check, time-travel test
ClinicalTrials.gov API	Every registered trial, with start dates	Evidence check, time-travel test
DailyMed	Official US drug labels	Safety check
openFDA	Adverse event reports	Safety check
PrimeKG is a large file (download it once on your laptop). The starter code includes a small sample graph so everything runs before you download it.
Record the PrimeKG build date. Write down which PrimeKG version you downloaded and when it was built, and store it with every result. The time-travel test depends on it.
The model, step by step
The task is link prediction: the graph has some "drug treats disease" links; the model learns to predict the missing ones. You build it in four stages, each one a baseline for the next.
1.	Explore the graph. Load PrimeKG, count node and edge types, trace known cases like sildenafil by hand. No ML yet.
2.	Path baseline. Score a drug by how many short paths connect it to the disease. Simple, explainable, surprisingly decent.
3.	Embedding baseline (TransE). Learn a vector for every node so that drug + "treats" lands near the disease. Teaches the core idea of knowledge graph embeddings.
4.	Relational graph neural network (R-GCN). Each node learns from its neighbors, with separate weights for each relation type. This is the main model.
Also build one more baseline that needs no learning: a popularity baseline that ranks drugs by how many trials and papers they already have (or by how many edges they have in the graph). If your model can't beat that, it's mostly picking famous drugs.
Training setup
•	Positive examples: known indication edges. Negative examples: random drug-disease pairs plus known contraindications.
•	Split by disease, not by edge, so the model is tested on diseases it hasn't seen treatments for.
•	When you hide a test link, also remove its reverse edge (PrimeKG stores both directions). If you only hide one direction, the model sees the answer through the flipped relation. Add a unit test that checks no test pair exists in either direction in the training graph.
•	Also run on TxGNN's disease splits so you can compare with their published numbers on the same setup.
•	Metrics: Hits@10 (is the right drug in the top 10?), mean reciprocal rank, and AUPRC.
•	Train on a free Colab or Kaggle GPU.
Explanations
•	For each prediction, find the paths that contributed most, using attention weights or a graph explainer.
•	Keep only paths of 2 to 4 steps that a person can read.
How we test it
Five tests, each answering a different doubt. Compare every model against the path baseline, TransE and the popularity baseline, and against TxGNN's published numbers on the same splits.
Test	How it works	What it proves
1. Standard	Hide 10 to 20% of known treatments (and their reverse edges); check they rank in the top 10	The model learned something real
2. No treatments seen	Hide every treatment for a set of diseases, using your own split and TxGNN's	It can help diseases with nothing approved
3. Case study replay	Take a known success such as sildenafil for PAH, delete its edge (both directions), retrain, and see where the drug ranks	It would have found a real repurposing
4. Time travel (your contribution)	Use the PrimeKG build date as the cutoff; check whether top ideas later appeared in trials or papers (details below)	Predictions point toward real future discoveries
5. Evidence check accuracy	Hand-label 150 abstracts; compare with the automatic labels	The automatic evidence step can be trusted
How the time-travel test works
PrimeKG is one snapshot, and most of its edges have no date, so you can't rebuild the graph as it was in an earlier year. The honest version is to treat the build date as the cutoff and look forward from there.
1.	Look up the build date of the PrimeKG version you downloaded. That is the cutoff.
2.	For each test disease, take the model's top 50 drugs.
3.	Drop any drug-disease pair that already had a trial or a paper before the cutoff. What's left is genuinely new.
4.	Check which of the remaining pairs got a trial (start date after the cutoff) or a paper after the cutoff.
5.	Report the hit rate next to two baselines: random drugs, and the most popular drugs (most trials and papers). Beating random only shows you can find famous drugs; beating the popularity baseline is the real result.
Say this limitation in the write-up in one plain sentence: the graph has no per-edge dates, so the cutoff is the snapshot date, not a rebuilt past. A reviewer will ask, and answering it first reads as careful.
Why the case study replay masks first. The known answer is probably already in the graph, so replaying it without hiding it would just return what the model was given. Masking the edge and retraining is what makes it a test.
Tech stack
Heavy ML runs offline; the app serves precomputed results fast; slow outside lookups run as background jobs. Everything is free.
Layer	Tool	Why
Graph in memory	Python, numpy CSR adjacency, pandas	Fast and light on 4M+ edges (already built)
Models	PyTorch, PyTorch Geometric	Standard for graph neural networks
Training	Free Google Colab or Kaggle GPU	No cost
Experiment tracking	MLflow	Every run's settings and scores logged and reproducible
Data versioning	Versioned PrimeKG snapshot + hash recorded with every result, including the build date	Provenance for every claim, and the cutoff for the time-travel test
API	FastAPI + Pydantic, OpenAPI docs	Typed endpoints; TypeScript client generated from the schema
Database	PostgreSQL (Neon)	Predictions, evidence, safety data, hypothesis notebook
Literature search	pgvector + a small biomedical sentence-embedding model	Semantic search over PubMed abstracts for the evidence check
Background jobs	RQ + Redis	Slow PubMed, trials and FDA calls run with retries, rate limits and caching
Outside data	PubMed E-utilities, ClinicalTrials.gov API, DailyMed, openFDA	Evidence and safety
Frontend	React + TypeScript + Tailwind, TanStack Query, Cytoscape.js	Interactive graph and research workspace
Quality	pytest, ruff, mypy, Vitest	Tests, lint and type checks (including the no-leakage test for splits)
Packaging + CI	Docker Compose, GitHub Actions	One-command setup; checks on every push
Hosting	Vercel (frontend), Render (API + worker), Neon (Postgres), free Redis tier	$0 to run
Deliberately skipped: a graph database such as Neo4j (its free cloud tier is too small for PrimeKG, and the in-memory graph is already fast), paid LLM APIs (embeddings plus a small classifier do the job, at $0 and fully reproducible), and Kubernetes or microservices (overkill here).
Schedule
About 120 hours for Version 1 at about 10 hours a week, with winter break covering the heavier modeling. The new items add roughly 8 hours; making signature reversal optional gives that time back.
Phase	Dates	Hours	What gets built	New in this version
1. Explore the graph	Oct 5 to Oct 18	~15	Load PrimeKG, graph stats, find paths between any two nodes, trace known repurposing cases (sildenafil) by hand	Record the PrimeKG build date and version hash
2. Path baseline + evaluation harness	Oct 19 to Nov 1	~15	Disease-based train/test split, Hits@10, MRR	Remove reverse edges of test links, with a unit test; popularity baseline; load TxGNN's splits
3. TransE embeddings	Nov 2 to Nov 15	~15	Knowledge graph embeddings, compared with both baselines	
4. Relational GNN	Nov 16 to Dec 6	~25	Train on Colab, tune, compare to all baselines	Run on TxGNN's splits
5. Explanations + evidence + safety checks	Dec 7 to Dec 20	~18	Reasoning paths, PubMed and trials labels, label warnings; hand-label 150 abstracts	Pick the focus disease area by Dec 7; line up one domain expert
6. Winter break: app	Dec 21 to Jan 3	~22	FastAPI, React, interactive graph, evidence brief	
7. Case studies, time-travel test, deploy, write-up	Jan 4 to Jan 17	~10	Mask-and-retrain case studies, time-travel test against both baselines, deploy	Expert reviews the top 10 for the focus disease; record the result
8. Version 2	Jan 18 to Feb 28	open	Watch list, paper reader, TxGNN comparison	
Expert review, in practice. Ask one person with a biology or pharmacology background (a PhD student, a faculty member, a lab you know) for 20 minutes. Give them the top 10 candidates and evidence briefs for the focus disease, and ask which they'd consider worth testing and why. Ask early, because people are slow to answer, and write down their role and what they said.
Skills, demo and resume
Skills you learn
Skill	Where
Graph data and graph algorithms	Exploring PrimeKG, path baseline
Knowledge graph embeddings	TransE
Graph neural networks	Relational GNN in PyTorch Geometric
ML evaluation done right	Disease-based splits, leakage checks, ranking metrics, popularity baseline, time-travel test
Explainable AI	Reasoning paths
Working with public health APIs	PubMed, ClinicalTrials.gov, DailyMed, openFDA
Biology basics	Drugs, targets, pathways, indications
Full stack + interactive graph UI	FastAPI, React, Cytoscape.js
Demo video (2 minutes)
1.	The problem in one line.
2.	Search a disease where the answer is known and watch the model find it (a masked case study).
3.	Click into a path.
4.	Show the evidence and safety check.
5.	Show the time-travel result next to the popularity baseline.
Resume bullets (fill in brackets with real numbers)
\resumeProjectHeading
  {\textbf{RepurposeMap} $|$ \emph{PyTorch Geometric, Python, FastAPI, React, PostgreSQL}}{Oct 2026 -- Present}
  \resumeItemListStart
    \resumeItem{Built an explainable drug repurposing tool on a 4M-edge biomedical knowledge graph, using a relational graph neural network to rank existing drugs for diseases with no approved treatment; Hits@10 of [X] vs [X] for the strongest baseline, evaluated on disease-held-out splits with reverse edges removed}
    \resumeItem{Added automated evidence and safety checks against PubMed, ClinicalTrials.gov and FDA label data, turning raw predictions into a triaged shortlist with one-page evidence briefs}
    \resumeItem{Designed a forward-looking evaluation showing [X] of the top [Y] predictions that had no prior studies later appeared in published studies or trials, vs [X] for a popularity baseline}
    \resumeItem{Had [N] of the top 10 candidates for [disease area] reviewed by a [role], who judged [X] worth testing}
  \resumeItemListEnd
Sources
•	Building a knowledge graph to enable precision medicine. Chandak, Huang, Zitnik, Scientific Data, 2023 (PrimeKG)
•	PrimeKG project page, Harvard Zitnik Lab
•	A foundation model for clinician-centered drug repurposing. Nature Medicine, 2024 (TxGNN)
•	TxGNN coverage, News-Medical

##

