# RepurposeMap ML: graph exploration and path-quality baseline

This folder holds the Python package. It is at **Milestone 3: proper disease-based evaluation**, on top of Milestone 2's path quality and drug-ranking baseline. The main README has the full picture. This file covers the package layout, the scoring rules, the baseline formula, the evaluation methodology, and the commands.

> **Research use only.** Every output here is a hypothesis for expert review, not a treatment recommendation. A path in the graph is not evidence of efficacy. The drug ranking is a non-ML baseline, not a trained model. The synthetic sample is invented data. Do not cite it as biology.

## Package layout

- `repurposemap/adapters/primekg.py`: parses PrimeKG's real `kg.csv` (12 columns) into the generic graph.
- `repurposemap/graph/`: generic loader, name search, bounded path search (`find_paths`, `find_paths_detailed`), and statistics. Path search is unchanged in this milestone.
- `repurposemap/scoring/relation_policy.py`: the relation policy. Classifies each PrimeKG edge into one of four categories, and builds the filtered view that search runs on.
- `repurposemap/scoring/hubs.py`: distinct-neighbour counts on allowed edges, and the hub penalty.
- `repurposemap/scoring/promiscuity.py`: distinct-target counts for drugs, and the drug-side promiscuity penalty.
- `repurposemap/scoring/path_quality.py`: the path score, its breakdown, deterministic ranking, and the "why this order" explanation.
- `repurposemap/ranking/drugs.py`: the drug-ranking baseline for one disease.
- `repurposemap/evaluation/baseline.py`: the Milestone 2 per-pair sanity check. Kept for quick manual checks; superseded for CLI use.
- `repurposemap/evaluation/indications.py`: extracts known drug-disease indication records from the graph.
- `repurposemap/evaluation/split.py`: `EvaluationConfig` and the deterministic disease-based train/test split.
- `repurposemap/evaluation/training_graph.py`: builds the one training graph with test diseases' treatments removed.
- `repurposemap/evaluation/leakage.py`: explicit leakage checks that fail loudly.
- `repurposemap/evaluation/metrics.py`: filtered rank, Hits@K, MRR. Decoupled from any one model.
- `repurposemap/evaluation/random_baseline.py`: the random-order comparison baseline.
- `repurposemap/evaluation/disease_eval.py`: ties the above into `evaluate_disease_split`, the function the CLI calls.
- `repurposemap/cli.py`: argument parsing and output only.

## Node identity

- **PrimeKG:** the node index from the file, as a string. Each node carries `name`, `node_type`, `external_id` and `provenance`.
- **Synthetic sample:** the entity name, which is also its `name`.

Names are never graph keys. A name resolves to an ID only if it matches exactly one node. Disease and drug lookups for ranking are limited to nodes of that type, so `pulmonary arterial hypertension` resolves to the disease node, not to the phenotype with the same name.

## Relation policy (PrimeKG)

PrimeKG has 30 relation labels and 33 `(relation, display_relation)` pairs. Every pair is classified in `PRIMEKG_RELATION_POLICY`, and the test suite checks that all 33 are present. The table below gives the edge count in each category, counted over PrimeKG's 8,100,498 stored rows. Each undirected link appears twice.

| Category | Rows | Pairs | Meaning |
|---|---:|---|---|
| preferred mechanistic | 921,024 | `drug_protein target`, `disease_protein associated with`, `protein_protein ppi`, `pathway_protein interacts with` | Direct molecular or disease-gene mechanism. Weight 1.0. |
| acceptable | 3,957,720 | `drug_protein enzyme / transporter / carrier`, `bioprocess_protein`, `cellcomp_protein`, `molfunc_protein`, `phenotype_protein`, `anatomy_protein_present`, `disease_phenotype_positive` | Real biology, but indirect, contextual or broad. Weight 0.5. |
| caution | 463,864 | `drug_effect side effect`, `disease_phenotype_negative`, `anatomy_protein_absent`, the five `exposure_*` pairs, and eight `*_* parent-child` ontology pairs | Weak, broad, adverse, or ontology hierarchy. Weight -0.5. |
| excluded | 2,757,890 | `drug_drug synergistic interaction`, `indication`, `contraindication`, `off-label use` | Not usable as path evidence. A path with one is invalid. |

Why these choices:

- **Drug-drug interactions are excluded.** They say nothing about a disease, and they are 33% of all rows. Keeping them would let paths run through other drugs.
- **Treatment labels are excluded.** `indication`, `contraindication` and `off-label use` are the answers being tested. Using them as path evidence would be circular.
- **Ontology hierarchy (`parent-child`) is penalised.** A parent term connects unrelated things, so these edges create most generic shortcuts.
- **Unmapped pairs are excluded**, so a relation added in a future PrimeKG release cannot count as evidence until someone decides how to classify it.

This is a transparent research heuristic. It is not a medically validated rule. Changing a category changes the rankings, so every change should be recorded here.

## Hub analysis (PrimeKG)

Measured on the full file.

Edge degree (every stored edge, all relations):

| Statistic | Value |
|---|---:|
| Median | 8 |
| 90th percentile | 272 |
| 95th percentile | 412 |
| 99th percentile | 1,979 |
| Maximum | 34,710 (`multi-cellular organism`, anatomy) |

Distinct neighbours over allowed edges (the measure used for the penalty):

| Statistic | Value |
|---|---:|
| Median | 4 |
| 90th percentile | 107 |
| 95th percentile | 165 |
| 99th percentile | 300 |
| Maximum | 17,355 (`multi-cellular organism`, anatomy) |

Highest-degree nodes by type, by raw edge degree:

- **anatomy:** multi-cellular organism (34,710); small intestine (33,710)
- **biological process:** regulation of transcription by RNA polymerase II (3,312)
- **cellular component:** nucleus (10,814); cytosol (10,572)
- **disease:** Mendelian disease (3,048); hereditary breast ovarian cancer syndrome (2,438)
- **drug:** quinidine (5,200), mainly through `drug_drug` edges. After the policy filter the highest drug has 341 neighbours (pregabalin).
- **effect/phenotype:** autosomal recessive inheritance (4,798); autosomal dominant inheritance (4,410)
- **gene/protein:** UBC (11,160); ETS1 (3,310); GATA2 (3,188)
- **molecular function:** protein binding (25,092); metal ion binding (4,658)
- **pathway:** neutrophil degranulation (958)

Generic hubs do create shortcuts. `protein binding` and `nucleus` are reachable from almost everything, so any path through them says little about a disease. The penalty below handles this by degree alone. No node is named in the code.

### Hub penalty

The penalty uses **distinct neighbours over allowed edges**. Drug-drug edges are not counted, so a drug does not look like a hub because of them.

- Distinct neighbours above **300** (the 99th percentile): penalty **0.25**
- Distinct neighbours above **2,000** (about the top 0.1%, 151 nodes): penalty **0.75**

The penalty applies only to **intermediate** nodes of a path. The two endpoints are the question being asked, so they are never penalised.

## Path-quality score

For a path with `L` hops and intermediate nodes `I`:

```
length_score   = 1 / L
relation_score = mean over hops of weight(category)
                 weights: preferred 1.0, acceptable 0.5, caution -0.5
hub_penalty    = sum over intermediate nodes of the tier penalty (0, 0.25 or 0.75)
total_score    = length_score + relation_score - hub_penalty
```

Rules:

- A path with any **excluded or unmapped** hop is **invalid**. It gets no total, and it is never ranked.
- Shorter paths score higher when relations are equal, because `1/L` falls with length.
- Ties are broken deterministically: shorter first, then by the sequence of node IDs and relation labels. Repeated runs give the same order.
- The score breakdown is a structured object with these keys: `valid`, `total_score`, `length_score`, `relation_score`, `hub_penalty`, `hop_categories`, `flags`. Flags name each caution hop, each hub node, and each invalid hop.

This is **not a confidence value**. It orders paths by simple, stated rules. It is not a probability, and it says nothing about whether the path is biologically true.

The explanation for the order names the components that differ. For example, `#1 ranks above #2: 2 hop(s) vs 4 hop(s); relation score 0.75 vs 1.00`.

## Drug-ranking baseline

This is a **research hypothesis ranking**. It is not a treatment recommendation and it is not a trained model.

Formula, for a drug `d` and a disease `D`:

1. **Candidates.** Drug nodes within `max_path_length` hops of `D` along allowed edges. A backward search finds them. If there are more than `max_candidates`, the nearest (fewest hops) are kept, ties by ID, and the cut is reported.
2. **Paths.** Directed paths `d -> ... -> D` of at most `max_path_length` hops (default **3**), found by bounded search. At most `paths_per_drug` (default **50**) of the shortest are kept. Search runs on the policy-filtered view, so excluded edges cannot fill the cap.
3. **Score each path** with the path-quality score. Invalid paths drop out.
4. **Path sum.** With valid path totals `s_1 >= s_2 >= ...`:

   ```
   path_sum(d) = sum of s_i for i = 1 .. min(3, n)
   ```

   where `n` is the number of valid paths. `K = 3` is `SCORED_PATHS_PER_DRUG`.
5. **Promiscuity penalty.** Count the distinct proteins linked to the drug by a `target` edge (`drug_protein` with display `target`). Enzyme, transporter and carrier roles are not counted.

   | Distinct targets | Penalty |
   |---|---:|
   | 26 or fewer (the 99th percentile of PrimeKG drugs) | 0 |
   | 27 to 100 | 0.25 |
   | more than 100 | 0.75 |

6. **Drug score.**

   ```
   drug_score(d) = path_sum(d) - promiscuity_penalty(d)
   ```

7. **Order.** Drug score (descending), then number of valid paths, then drug name, then ID. Drugs with no valid path are not ranked.

Output for each drug: rank, drug ID, drug name, drug score, number of valid paths, number of targets, any promiscuity penalty, and the best path with its breakdown.

Why the penalty exists. Each target is another chance to reach a disease gene, so a drug with many targets gets more paths almost by construction. Before the penalty, fostamatinib (302 targets), copper (142) and zinc (124) ranked near the top of the PAH list largely because of their target counts. The hub penalty applies only to intermediate nodes, so it could not catch these drug endpoints. The count uses targets, not allowed neighbours, because side-effect edges would inflate neighbour counts for drugs such as sildenafil and imatinib, which are not promiscuous.

Known limits of the formula:

- The path sum saturates at **4.5** (three preferred two-hop paths at 1.5 each). Many drugs tie near that ceiling, so the ranking discriminates less than it looks.
- The penalty is a fixed subtraction. A drug with many targets can still rank high if its path sum is large enough.

## Evaluation harness

`evaluate-baseline` is a **disease-based** evaluation. It measures whether the baseline can recover a known treatment after every treatment for that *disease* is hidden, not after hiding one drug-disease edge while leaving everything else about that disease visible.

### Why not split by edge

An edge-level split hides one `(drug, disease)` pair at a time and leaves the rest of the graph untouched. That is safe for a baseline that cannot learn from the graph, which is what Milestone 2's sanity check did. It is not safe for anything that fits on the graph, such as a future TransE or R-GCN model:

- The disease keeps every one of its *other* drug indications visible. A model trained on "all edges except this one" can still learn the disease's treatment profile from its other treatments, and from drugs that resemble the held-out one.
- Re-running the hide-one-pair-at-a-time process for every pair never produces one single training graph. A learned model needs exactly one training graph to fit, so the evaluation has to decide, once, what is hidden for that one fit.

### The split

The split is on **disease ID**:

1. Extract every `(drug, disease)` pair linked by an `indication` edge (`repurposemap.evaluation.indications`). Contraindications, drug-drug interactions, and weak graph associations (gene, phenotype, anatomy links, and the rest) are never treated as treatments.
2. Choose a set of **test diseases**, deterministically for a given seed: `round(test_fraction * n_diseases)`, capped by `max_test_diseases`, at least one (`repurposemap.evaluation.split`).
3. **Every** indication edge of a test disease goes to the test set. **None** of it goes to training. A disease with three known drugs has all three held out together, never some in training and some in test.
4. Build one training graph: a view of the full graph with every test disease's indication edges removed, in both directions (`repurposemap.evaluation.training_graph`). Everything else stays, including gene, protein, pathway, phenotype and anatomy links for test diseases, and the indication edges of train diseases. This simulates "a disease with its biological knowledge, but no treatment information" for every test disease.

### What stays visible for a test disease

Only its `indication` edges are gone. Its disease-gene associations, phenotypes, anatomy links, and position in the disease hierarchy all stay. The path baseline could, in principle, use any of that; the relation policy (not the split) decides what actually counts as evidence.

### Leakage checks

`repurposemap.evaluation.leakage.assert_no_leakage` runs after the training graph is built and before any ranking. It fails loudly (raises `ValueError`, listing every violation) if:

1. Any indication edge in the training graph touches a test disease.
2. Any held-out `(drug, disease)` test pair is still connected by an indication edge.
3. The train and test disease sets overlap.

A fourth risk, that the ranking code itself inspects a held-out edge, is handled structurally rather than checked at run time: the ranking only ever sees the training graph (the edges are gone, not hidden behind a per-call argument), and the relation policy excludes the `indication` relation everywhere regardless.

### Metrics

Computed over every held-out `(drug, disease)` pair, not over diseases: a disease with three true drugs contributes three ranks.

- **Filtered rank.** For a disease with several true drugs, finding one drug's rank first removes every *other* true drug for that disease from the ranked list. This is the standard knowledge-graph link-prediction practice, so that a baseline ranking two correct drugs 1st and 2nd is not punished for the second one "pushing down" the first.
- **Hits@K** (K = 1, 3, 10): the share of held-out pairs whose drug's filtered rank is at most K. 1.0 means every true drug was in the top K; 0.0 means none were.
- **MRR (mean reciprocal rank):** the mean of `1/rank` over every held-out pair, with 0 for a pair that is not ranked at all. It rewards ranking the true drug first much more than ranking it tenth, and it is never dominated by one disease with many candidates the way a disease-averaged metric could be.

### Random baseline

Each test disease also gets a **random-order baseline**: the same candidate drugs the path baseline found for that disease, shuffled with a seed derived from `(seed, disease_id)`. Same candidates, same metrics, different order. This is the real question the evaluation asks: not "can the baseline rank the true drug highly", but "does the baseline do any better than shuffling the same list". A result that does not beat random is not evidence the baseline captures anything.

### Configuration

Every knob is one `EvaluationConfig` (`repurposemap.evaluation.split`), with no hidden defaults: `seed`, `test_fraction`, `max_test_diseases`, `top_ks`, and the path-ranking parameters (`max_path_length`, `paths_per_drug`, `max_candidates`, `time_limit_per_drug_s`, `max_edge_checks_per_drug`). `evaluate-baseline`'s output prints the configuration it ran with, so a result is never read without knowing what produced it. `max_candidates` defaults to 500 for evaluation, not rank-drugs' 2,000, because evaluation runs this once per test disease and must stay well under an hour for a few dozen diseases.

### What this does not show

- This is still a **sanity check on a handful of diseases**, not a validated benchmark. Twenty test diseases cannot support a claim of predictive performance.
- A pair the baseline does not rank scores 0 on every metric, which conflates "not reachable within the hop and candidate limits" with "the baseline ranks it poorly".
- The old per-pair sanity check (`repurposemap.evaluation.evaluate_baseline`) still exists as a function, for a quick manual check of one or two pairs. The CLI no longer uses it; `evaluate-baseline` now runs the disease-based evaluation above.

## Measured results (full PrimeKG, this machine)

Machine: 8-core Mac, 8 GB RAM. Times are wall-clock.

- **Load** (`load_primekg`): 94 to 99 seconds.
- **Relation policy and hub counts:** computed lazily, only for nodes on a path.
- **explain-paths** (one pair, 200 candidate paths): 9 to 12 seconds after load.
- **rank-drugs** for pulmonary arterial hypertension: 53 to 60 seconds, 2,000 candidates (cap reached, about 30 ms per drug).
- **evaluate-baseline**, 20 test diseases, the current defaults (500 candidates, 5 s, 1,000,000 edge checks per drug): about 18.5 minutes in total, roughly 55 seconds per disease on average. See "Disease-based evaluation: a controlled benchmark" below for how this number was reached; an earlier, unfixed version of this code path took far longer, for a reason that had nothing to do with the diseases themselves.

Known-pair sanity checks. Each direct indication edge is the raw shortest path, and the policy excludes it. These are sanity checks only, not evidence of performance.

- **Sildenafil -> pulmonary arterial hypertension.** Raw: 1-hop direct indication (invalid). Quality: the top three valid paths all use a `side effect` hop (caution) and tie at 0.667. The caution penalty ranks them, but it cannot make them disappear. Biological content is weak.
- **Metformin -> type 2 diabetes mellitus.** Raw: direct indication (invalid), then a 2-hop transporter route through SLC22A3 (valid). Quality: that route ranks first at 1.250. It ties with several 4-hop target routes, which score 1.250 by length and relation.
- **Imatinib -> chronic myelogenous leukemia, BCR-ABL1 positive.** Raw: direct indication (invalid), then 2-hop routes through ABL1 and BCR. Both carry the hub penalty of 0.25 (753 and 312 allowed neighbours). Quality: a 3-hop target route through DDR1 and PER1 ranks first at 1.333 with no hub penalty, above both 2-hop routes.

Drug-ranking example, pulmonary arterial hypertension, top 10 after the promiscuity penalty (research hypothesis ranking, not a recommendation):

1. Dalfampridine, score 4.167, 8 valid paths, 16 targets. Best path: target KCNA5, which PrimeKG links to PAH.
2. Resveratrol, score 4.000, 26 valid paths, 26 targets.
3. Ascorbic acid, score 4.000, 8 valid paths, 21 targets.
4. Dibotermin alfa, score 3.917, 10 valid paths, 2 targets. Best path: target BMPR2, which PrimeKG links to PAH.
5. ATP, score 3.917, 23 valid paths, 42 targets, penalty 0.25. Best path: target ACVRL1.
6. Fostamatinib, score 3.750, 50 valid paths, 302 targets, penalty 0.75. Best path: target EIF2AK4. Before the penalty it ranked first at 4.500.
7. Tamoxifen, score 3.750, 33 valid paths, 17 targets.
8. Belinostat, score 3.750, 21 valid paths, 11 targets. Best path: target HDAC1.
9. Panobinostat, score 3.750, 21 valid paths, 11 targets. Best path: target HDAC1.
10. Myristic acid, score 3.750, 10 valid paths, 21 targets.

Before the penalty, copper, zinc and zinc acetate were in the top ten. Copper and zinc acetate are now outside the top 15, and zinc chloride is 15th with a 0.25 penalty. The list still includes some drugs with a few targets, such as dorsomorphin and heptyl glucoside, which score on path count and path length. These are candidates for review, not findings.

### Disease-based evaluation: a controlled benchmark

Run with `evaluate-baseline --source primekg --test-diseases 20 --seed 42`, the current defaults (500 candidates, 5 second and 1,000,000-edge-check per-drug budget), on a single process with nothing else competing for the machine:

- **20 test diseases, 147 held-out (drug, disease) pairs.**

| Metric | Path baseline | Random baseline |
|---|---:|---:|
| Hits@10 | **0.034** | **0.007** |
| MRR | **0.0158** | **0.0035** |

This is a **small, controlled evaluation**, not a validated benchmark: 20 diseases and 147 pairs. Within that small sample, the path baseline beats the random-order baseline by about **5x on Hits@10** and about **4.5x on MRR**. That is a real, reproducible signal that quality-ranked paths carry more information than an arbitrary order over the same candidates. It is not large, and **absolute performance remains weak**: Hits@10 of 0.034 means the true drug lands in the top 10 for roughly 1 in 30 held-out pairs.

This baseline, and this evaluation harness, exist as a **comparison point for TransE and R-GCN**, planned for later milestones (see Roadmap). Whatever those models score, it should be read against this baseline and against the random-order baseline on the same split and the same metrics.

**These results are not clinical evidence.** They measure whether a heuristic path score can recover a known treatment after it is hidden from the graph. They say nothing about whether any ranked drug would work in a patient, and must not be read as if they did.

A controlled benchmark, run before this number, found that an early, unfixed version of the per-disease ranking call built two nested graph views where one would do, which made the search pay the cost of filtering every edge twice. Fixing that (folding the filter into one view, verified on a fast and a slow disease to give byte-identical rankings, 2.8x to 3.1x faster) is what makes a 20-disease run take minutes rather than tens of minutes to over an hour; it did not change any rank, score, or path.

## Commands

Run from the repository root. The scoring commands need `--source primekg`.

```bash
# existing exploration commands
python -m repurposemap stats --source primekg
python -m repurposemap search --source primekg --query "sildenafil"
python -m repurposemap paths --source primekg --source-name "sildenafil" --target-id 38436 --max-path-length 2

# raw shortest paths next to quality-ranked paths
python -m repurposemap explain-paths --source primekg \
  --source-name "sildenafil" --target-id 38436 --max-results 5

# drug-ranking baseline (research hypothesis ranking)
python -m repurposemap rank-drugs --source primekg \
  --disease "pulmonary arterial hypertension" --top-k 10
python -m repurposemap rank-drugs --source primekg --disease-id 38436 --top-k 10

# disease-based evaluation, with a random-order baseline for comparison
python -m repurposemap evaluate-baseline --source primekg --test-diseases 20 --seed 42
```

Useful options:

- `explain-paths`: `--max-path-length` (default 4), `--max-results` (raw and ranked paths shown, default 5), `--candidate-paths` (default 200), `--time-limit`.
- `rank-drugs`: `--max-path-length` (default 3), `--paths-per-drug` (default 50), `--max-candidates` (default 2000), `--time-limit-per-drug` (default 5 seconds), `--max-edge-checks-per-drug` (default 1,000,000).
- `evaluate-baseline`: `--test-diseases` (cap on test diseases, default 20), `--test-fraction` (share of diseases held out before that cap, default 0.1), `--seed` (default 0), plus the same path-ranking limits as `rank-drugs` except `--max-candidates` defaults to 500 here, to keep a multi-disease run fast.

## Limitations

- **Connectivity is not proof of efficacy.** A path that exists in the graph, however well scored, is a hypothesis.
- **The path score is heuristic.** It is not validated against any gold standard, and the weights were chosen by reasoning, not fitted.
- **The baseline is not a trained model.** No machine learning exists in this milestone.
- **The evaluation is still small.** 20 test diseases, 147 held-out pairs (about 1.5% of PrimeKG's diseases with a known indication) cannot support any claim of clinical or predictive performance, even with a disease-based split and a random baseline beaten by 4.5x to 5x. It rules out the most obvious form of leakage and shows the baseline does better than chance on this sample; it does not validate the baseline.
- **A drug beyond the candidate cap looks unranked**, which is not the same as the baseline ranking it poorly. `max_candidates` defaults to 500 for evaluation (versus 2,000 for `rank-drugs`), and `DrugRanking.candidate_cap_hit` / `TreatmentOutcome.candidate_cap_hit` say when that happened.
- **Promiscuous drugs are penalised by a fixed amount.** The penalty removes the worst of the generic compounds from the top, but a drug with many targets can still rank high if its path sum is large.
- **Low-target drugs can rank on path count.** Drugs with two to four targets can have several short valid paths, so they rank well for reasons that are not mechanistic.
- **The score saturates** at 4.5 with K = 3, so many drugs tie.
- **Search limits shape the result.** Candidates are capped at 2,000 (the nearest first), and paths per drug at 50. A drug beyond the cap is reported as not ranked, not as a poor drug.
- **Shortest-first path search.** Each drug keeps its 50 shortest paths, so its score can change if the cap changes.
- **The relation policy is a heuristic**, written by hand, and may disagree with domain experts.
- **Sample data is not scored.** The synthetic sample uses relation names outside the PrimeKG policy, so every path there is invalid. The scoring commands require `--source primekg`.
- **Memory.** Loading the full graph needs about 6 GB free. The tests do not load it.

## Tests

```bash
pytest
```

215 tests run in about 1.5 seconds. They use the synthetic sample, the 15-row fixture in `tests/fixtures/`, and tiny graphs in the PrimeKG schema built in `tests/graph_builders.py`. None of them need the full PrimeKG file. The scoring tests cover relation scoring, hub tiers, length and ordering, the breakdown structure, and synthetic compatibility through a custom policy. The ranking tests cover score formula, determinism, candidate cap, leakage prevention, Hits@10, reciprocal rank, and the backward-search edge-check budget. `tests/test_disease_evaluation.py` covers indication extraction, the disease-based split (determinism, the test-fraction and cap, disjointness, validation), the training graph, every leakage check (including one that fails on purpose), the metrics (filtered rank, Hits@K, MRR), the random baseline, an end-to-end run with a disease that has two true drugs and one that cannot be ranked at all, and that a disease's search-completeness fields (candidate cap, per-drug truncation, backward-search truncation) are exposed on its outcome. The CLI tests cover all three scoring commands.

If `python -m repurposemap` cannot find the package on macOS, see the troubleshooting section in the main README.
