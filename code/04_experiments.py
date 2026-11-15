import sys, json, time
import numpy as np
sys.path.insert(0, "code")
from lib import Data, topn, rerank_cap, label

P = sys.argv[1] if len(sys.argv) > 1 else "data/prepared"
D = Data(P)
n = len(D.eligible); rows = np.arange(len(D.users))
head1 = np.zeros(n, bool); head1[D.head_rank[: int(round(n * 0.01))]] = True
CA = D.tags["CA"]
caps = [10, 8, 6, 5, 4, 3, 2, 1, 0]
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
configs = {}
def add(name, L): configs.setdefault(name, []).append(L)

def ca_slot(L, sc):
    L = L.copy()
    has = CA[L].any(1)
    s2 = np.where(CA[None, :], sc, -np.inf)
    best = s2.argmax(1)
    ok = (~has) & np.isfinite(s2[np.arange(len(L)), best])
    L[ok, -1] = best[ok]
    return L

t0 = time.time(); B = 1500
logpop = np.log(D.pop.astype(np.float64))
for s in range(0, len(rows), B):
    r = rows[s:s + B]
    sc = D.personalized_scores(r)
    for c in caps:
        add(f"cap{c}", topn(sc) if c == 10 else rerank_cap(sc, head1, c))
    pos = np.where(np.isfinite(sc), sc, 0)
    for a in alphas:
        s2 = np.where(np.isfinite(sc) & (sc > 0), np.log(np.maximum(pos, 1e-12)) - a * logpop[None, :], -np.inf)
        add(f"pen{a:.1f}", topn(s2))
    add("ca_slot", ca_slot(topn(sc), sc))
    add("cap3_ca_slot", ca_slot(rerank_cap(sc, head1, 3), sc))
    print(s + len(r), f"{time.time()-t0:.0f}s", flush=True)

rng = np.random.default_rng(11)
BOOT = rng.integers(0, len(rows), size=(1000, len(rows)))
n_hold_el = np.asarray(D.H.sum(axis=1)).ravel()
base_hits = None
hold_share = {t: float(m[D.H.nonzero()[1]].mean()) for t, m in D.tags.items()}
res = {}
for name in ["cap10"] + [k for k in configs if k != "cap10"]:
    L = np.vstack(configs[name]); np.save(f"{P}/lists_{name}.npy", L)
    m, hits = label(D, L)
    rec = hits.sum() / n_hold_el.sum()
    bs = hits[BOOT].sum(1) / n_hold_el[BOOT].sum(1)
    m["recall_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    if base_hits is None: base_hits = hits
    d = (hits[BOOT].sum(1) - base_hits[BOOT].sum(1)) / n_hold_el[BOOT].sum(1)
    m["recall_delta_vs_personalized"] = float((hits.sum() - base_hits.sum()) / n_hold_el.sum())
    m["recall_delta_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    for t in D.tags:
        m[f"{t}_heldout_share"] = hold_share[t]
        m[f"{t}_demand_ratio"] = m[f"{t}_slot_share"] / hold_share[t]
    res[name] = m
json.dump(res, open("results/experiments.json", "w"), indent=2)
keys = ["recall@10_eligible_holdout", "recall_delta_vs_personalized", "pop_pct_mean", "head1_share", "shelf_used",
        "genre_spread", "share_1990s", "CA_slot_share", "CA_demand_ratio", "CA_screens_with_any", "FR_demand_ratio"]
print("config".ljust(14) + " ".join(k[:12].rjust(12) for k in keys))
for k, v in res.items():
    print(k.ljust(14) + " ".join(f"{v[x]:12.4f}" if isinstance(v[x], float) else f"{v[x]:12d}" for x in keys))
