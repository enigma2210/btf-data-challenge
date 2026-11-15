import sys, json
import numpy as np
sys.path.insert(0, "code")
from lib import Data, topn, label

P = sys.argv[1] if len(sys.argv) > 1 else "data/prepared"
test_users = Data(P).users
V = Data(P, seed=99, exclude=test_users)
assert len(np.intersect1d(V.users, test_users)) == 0
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
logpop = np.log(V.pop.astype(np.float64)); lists = {a: [] for a in alphas}
rows = np.arange(len(V.users))
for s in range(0, len(rows), 1500):
    r = rows[s:s + 1500]; sc = V.personalized_scores(r); pos = np.where(np.isfinite(sc), sc, 0)
    for a in alphas:
        s2 = np.where(np.isfinite(sc) & (sc > 0), np.log(np.maximum(pos, 1e-12)) - a * logpop[None, :], -np.inf)
        lists[a].append(topn(s2))
out = {}
for a in alphas:
    m, _ = label(V, np.vstack(lists[a])); out[f"{a:.1f}"] = m
base = out["0.0"]["recall@10_eligible_holdout"]
chosen = max(a for a in alphas if out[f"{a:.1f}"]["recall@10_eligible_holdout"] >= base)
json.dump({"validation": out, "rule": "largest alpha with validation recall >= alpha 0", "chosen_alpha": chosen},
          open("results/validation_alpha.json", "w"), indent=2)
for a in alphas:
    m = out[f"{a:.1f}"]
    print(a, round(m["recall@10_eligible_holdout"], 4), m["shelf_used"], round(m["head1_share"], 3), round(m["CA_slot_share"], 4))
print("chosen alpha:", chosen)
