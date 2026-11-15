import sys, json
import numpy as np
sys.path.insert(0, "code")
from lib import Data, topn, rerank_cap, label

P = sys.argv[1] if len(sys.argv) > 1 else "data/prepared"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
KS = [10, 8, 7, 6, 5, 4, 3]

def run(D):
    n = len(D.eligible); head1 = np.zeros(n, bool); head1[D.head_rank[: int(round(n * 0.01))]] = True
    logpop = np.log(D.pop.astype(np.float64)); rows = np.arange(len(D.users))
    lists = {}
    for s in range(0, len(rows), 1500):
        r = rows[s:s + 1500]; sc = D.personalized_scores(r); pos = np.where(np.isfinite(sc), sc, 0)
        for a in ALPHAS:
            s2 = sc if a == 0 else np.where(np.isfinite(sc) & (sc > 0), np.log(np.maximum(pos, 1e-12)) - a * logpop[None, :], -np.inf)
            for k in KS:
                L = topn(s2) if k == 10 else rerank_cap(s2, head1, k)
                lists.setdefault((a, k), []).append(L)
    return {key: np.vstack(v) for key, v in lists.items()}

test = Data(P)
val = Data(P, seed=99, exclude=test.users)
VL = run(val)
vm = {key: label(val, L)[0] for key, L in VL.items()}
base_v = vm[(0.0, 10)]["recall@10_eligible_holdout"]
ok = [key for key in vm if vm[key]["recall@10_eligible_holdout"] >= base_v]
chosen = min(ok, key=lambda key: vm[key]["head1_share"])
print("validation baseline", round(base_v, 4), "chosen", chosen, {k: round(vm[chosen][k], 4) for k in ["recall@10_eligible_holdout", "head1_share"]})

TL = run(test)
n_hold = np.asarray(test.H.sum(axis=1)).ravel()
rng = np.random.default_rng(11); BOOT = rng.integers(0, len(test.users), size=(1000, len(test.users)))
held = test.H.nonzero()[1]
base_m, base_hits = label(test, TL[(0.0, 10)])
out = {"rule": "lowest head-share with validation recall >= today's", "chosen": {"alpha": chosen[0], "K": chosen[1]},
       "validation": {f"a{a}_K{k}": {"recall": vm[(a, k)]["recall@10_eligible_holdout"], "head1_share": vm[(a, k)]["head1_share"]} for (a, k) in vm},
       "test": {}}
for (a, k), L in TL.items():
    m, hits = label(test, L)
    d = (hits[BOOT].sum(1) - base_hits[BOOT].sum(1)) / n_hold[BOOT].sum(1)
    m["recall_delta_vs_personalized"] = float((hits.sum() - base_hits.sum()) / n_hold.sum())
    m["recall_delta_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    bs = hits[BOOT].sum(1) / n_hold[BOOT].sum(1)
    m["recall_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for t, msk in test.tags.items():
        m[f"{t}_heldout_share"] = float(msk[held].mean()); m[f"{t}_demand_ratio"] = m[f"{t}_slot_share"] / m[f"{t}_heldout_share"]
    out["test"][f"a{a}_K{k}"] = m
    if (a, k) == chosen: np.save(f"{P}/lists_combo.npy", L)
json.dump(out, open("results/combined_lever.json", "w"), indent=2)
print("TEST (alpha, K): recall  delta [CI]  head  films  CA_slots")
for (a, k) in sorted(TL):
    m = out["test"][f"a{a}_K{k}"]
    if k in (10, 5, 3) or (a, k) == chosen:
        print((a, k), f'{m["recall@10_eligible_holdout"]:.4f} {m["recall_delta_vs_personalized"]:+.4f} [{m["recall_delta_ci95"][0]:+.4f},{m["recall_delta_ci95"][1]:+.4f}] {m["head1_share"]:.3f} {m["shelf_used"]} {m["CA_slot_share"]:.4f}', "<= chosen" if (a, k) == chosen else "")
