import sys, json
import numpy as np
sys.path.insert(0, "code")
from lib import Data

P = sys.argv[1] if len(sys.argv) > 1 else "data/prepared"
D = Data(P)
exp = json.load(open("results/experiments.json"))
rep = json.load(open("results/phase1_reproduction.json"))
val = json.load(open("results/validation_alpha.json"))
prep = json.load(open(f"{P}/prepare_stats.json"))
strict = None
try:
    strict = json.load(open("results/strict_tags.json"))
except FileNotFoundError:
    pass

def pick(m):
    keys = ["recall@10_eligible_holdout", "recall_ci95", "recall_delta_vs_personalized", "recall_delta_ci95",
            "pop_pct_mean", "head1_share", "shelf_used", "shelf_used_share", "slot_gini", "share_1990s",
            "genre_spread", "CA_catalog_share", "CA_slot_share", "CA_heldout_share", "CA_demand_ratio",
            "CA_exposure_ratio", "CA_screens_with_any", "FR_catalog_share", "FR_slot_share", "FR_heldout_share",
            "FR_demand_ratio", "FR_exposure_ratio", "FR_screens_with_any"]
    return {k: m.get(k) for k in keys}

lists = {k: np.load(f"{P}/lists_{k}.npy") for k in ["cap10", "combo", "pen0.4", "cap5", "ca_slot"]}
CA, FR = D.tags["CA"], D.tags["FR"]
ntrain = np.asarray(D.U.sum(1)).ravel()
cand = []
for u in range(len(D.users)):
    b = lists["pen0.4"][u]
    hit = D.H[u, b].sum()
    if 25 <= ntrain[u] <= 120 and CA[b].any() and hit >= 1 and D.H[u, lists["cap10"][u]].sum() >= 1:
        cand.append(u)
demo_u = cand[:: max(1, len(cand) // 4)][:4]
titles = D.mv.title.values
def item(i, u):
    return {"t": str(titles[i]), "pct": round(float(D.pop_pct[i]), 1), "ca": bool(CA[i]), "fr": bool(FR[i]),
            "hit": bool(D.H[u, i] > 0), "head": bool(i in set(D.head_rank[:110]))}
demos = []
for k, u in enumerate(demo_u):
    hist = D.U[u].indices
    recent = [str(titles[i]) for i in hist[np.argsort(-D.pop[hist])][:5]]
    demos.append({"label": f"Viewer {chr(65+k)}", "n_history": int(ntrain[u]), "history_sample": recent,
                  "screens": {name: [item(i, u) for i in L[u]] for name, L in lists.items()}})

out = {
    "meta": {"users_test": len(D.users), "eligible_films": len(D.eligible), "split": prep,
             "head1_films": 110, "validation_rule": val["rule"], "chosen_alpha": val["chosen_alpha"]},
    "reproduction": {k: pick(v) | {"recall": v["recall@10_eligible_holdout"]} for k, v in rep.items()},
    "configs": {k: pick(v) for k, v in exp.items()},
    "validation": {a: {"recall": m["recall@10_eligible_holdout"], "shelf_used": m["shelf_used"],
                       "head1_share": m["head1_share"]} for a, m in val["validation"].items()},
    "strict_tags": None if strict is None else {k: {a: b for a, b in v.items() if a.startswith("CA_")} for k, v in strict.items() if not k.startswith("_")},
    "strict_counts": None if strict is None else strict["_counts"],
    "demos": demos,
    "grid": {k: pick(v) for k, v in json.load(open("results/combined_lever.json"))["test"].items()},
    "combo_choice": json.load(open("results/combined_lever.json"))["chosen"],
    "phase1": {
        "gdelt": {"articles": 671681, "window": "11-17 Oct 2026", "top6_share": 0.505, "us_ratio": 6.2,
                   "africa_ratio": 0.32, "asia_ratio": 0.46, "china_india_people": 0.353, "china_india_mentions": 0.084},
        "mind": {"screens": 2609219, "slots": 97592931, "top293_slot_share": 0.373, "top293_catalog_share": 0.0028,
                  "slot_gini": 0.93, "taste_pp_min": 4.2, "taste_pp_max": 5.5, "novelty_short": 0.91, "novelty_heavy": 0.70},
    },
}
json.dump(out, open("docs/data.json", "w"), indent=1)
print("demos:", [d["label"] for d in demos], "candidates", len(cand))
