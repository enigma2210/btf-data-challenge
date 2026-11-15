# The Feed Balance Label

**Behind the Feed Data Challenge · Phase 2 · Team EpicDevs, University of New Brunswick**

A printable label, a Head Cap and an accuracy ledger for recommender systems. The label reports what a ranked list actually puts on the screen against baselines that already exist in the data (population, the catalog, the site's own menu, and what viewers go on to like), and the ledger reports what each fix costs in accuracy. It never needs the platform's model or source code.

**Live dashboard:** https://enigma2210.github.io/btf-data-challenge/

## Results

MovieLens 32M, 12,000 test viewers, same data split as our Phase 1 submission.

| | Today | **Team setting: Head Cap 5 + penalty α 0.2** | Penalty α 0.4 |
|---|---|---|---|
| Accuracy (recall@10) | 7.63% | 7.69% (change +0.06 pts, 95% CI −0.13 to +0.27) | 7.86% (change +0.24 pts, 95% CI +0.05 to +0.43) |
| Slots taken by the top 1% of films | 87.0% | 45.5% | 59.4% |
| Films ever shown (of 10,952) | 1,114 | 2,283 | 3,885 |
| Canadian films, share of slots (share of next likes: 2.06%) | 0.07% | 0.36% | 0.64% |
| Screens showing any Canadian film | 0.56% | 3.0% | 4.7% |

* The team setting was chosen from 42 combinations of Head Cap and popularity penalty on a separate set of 12,000 validation viewers, using a rule fixed in advance: the lowest top-1% share whose validation accuracy is not below today's.
* Strict check (Canada the only country of origin, 92 films): 0.21% of next likes but 0.01% of slots today (0.03×), 0.16× with the team setting.
* Blockbuster Test: genre spread rates an all-blockbuster list (0.80) as more diverse than the team setting (0.76), so the label reports the top-1% share instead.
* Phase 1 reproduces: 15,938,231 liked ratings, 176,028 viewers, 10,952 films; accuracy 7.63% (Phase 1: 7.6%), most-rated 4.59% (4.6%), genre cap 7.17% (7.2%).

## Repository

| Path | Contents |
|---|---|
| `code/` | Data preparation, recommender, re-ranking experiments, validation, Canadian-content audit, dashboard export |
| `results/` | All computed results (JSON) |
| `docs/` | Label Lab dashboard (GitHub Pages) |
| `dashboard/` | Dashboard template |
| `data/` | Wikidata tags for Canadian and French-language films |

## Definitions

* **Accuracy (recall@10):** share of each viewer's five held-out liked films that appear in their top 10.
* **Top-1% share:** share of top-10 slots taken by the 110 most-liked of 10,952 eligible films.
* **Head Cap K:** at most K of 10 slots may go to top-1% films.
* **Popularity penalty α:** each score is divided by popularity^α before ranking.
* **Demand ratio:** a group's share of slots divided by its share of viewers' next likes.

## Limitations

Top-10 lists are simulated from rating logs, not shown to people. MovieLens viewers are not a Canadian audience. Wikidata country of origin is a proxy for Canadian content, not CAVCO certification. No age, gender, race or politics exists in these data and none is inferred.

## Data

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4). MovieLens is used under the GroupLens research licence and is not redistributed here; no endorsement is implied. Wikidata content is CC0. Phase 1 sources: GDELT GKG 2.0, World Bank WDI, Microsoft MIND.

## Team

Rishabh Singh · Aayush Kharbanda · Randeep Sidhu
