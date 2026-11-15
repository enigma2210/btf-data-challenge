import sys, json, time
import numpy as np
sys.path.insert(0, "code")
from lib import Data, topn, rerank_cap, label

D = Data(sys.argv[1] if len(sys.argv) > 1 else "data/prepared")
rows = np.arange(len(D.users)); rng = np.random.default_rng(7)
t0 = time.time(); res = {}; B = 1500
lists = {k: [] for k in ["most_rated", "personalized", "genre_cap3", "random"]}
for s in range(0, len(rows), B):
    r = rows[s:s + B]
    lists["most_rated"].append(topn(D.popularity_scores(r)))
    ps = D.personalized_scores(r)
    lists["personalized"].append(topn(ps))
    gm = D.G[:, :][:, D.dom_genre[r]].T > 0
    lists["genre_cap3"].append(rerank_cap(ps, gm, 3))
    lists["random"].append(topn(D.random_scores(r, rng)))
    print(s + len(r), f"{time.time()-t0:.0f}s", flush=True)
for k, v in lists.items():
    L = np.vstack(v); np.save(f"{D.P}/lists_{k}.npy", L)
    res[k], _ = label(D, L)
json.dump(res, open("results/phase1_reproduction.json", "w"), indent=2)
for k, v in res.items():
    print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
