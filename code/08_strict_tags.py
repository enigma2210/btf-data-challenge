import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "code")
from lib import Data

P = sys.argv[1] if len(sys.argv) > 1 else "data/prepared"
CSV = sys.argv[2] if len(sys.argv) > 2 else "data/wikidata_ca_fr_strict.csv"
D = Data(P)
wd = pd.read_csv(CSV)
norm = lambda s: "tt" + str(s)[2:].lstrip("0")
imdb = D.mv.imdb.map(norm).values
tags = {"CA": np.isin(imdb, [norm(x) for x in wd.loc[wd.tag == "CA_ONLY", "imdb"]]),
        "FR": np.isin(imdb, [norm(x) for x in wd.loc[wd.tag == "FR_ONLY", "imdb"]])}
held = D.H.nonzero()[1]
import os
names = list(json.load(open("results/experiments.json")).keys()) + ["most_rated", "personalized", "genre_cap3", "random"] + (["combo"] if os.path.exists(f"{P}/lists_combo.npy") else [])
out = {"_counts": {t: int(m.sum()) for t, m in tags.items()},
       "_top_titles": {t: list(D.mv.title.values[m][np.argsort(-D.pop[m])][:15]) for t, m in tags.items()}}
for k in names:
    L = np.load(f"{P}/lists_{k}.npy"); flat = L.ravel(); r = {}
    for t, m in tags.items():
        cat, hs, sl = m.mean(), m[held].mean(), m[flat].mean()
        r.update({f"{t}_catalog_share": float(cat), f"{t}_heldout_share": float(hs), f"{t}_slot_share": float(sl),
                  f"{t}_exposure_ratio": float(sl / cat), f"{t}_demand_ratio": float(sl / hs) if hs else None,
                  f"{t}_screens_with_any": float((m[L].sum(1) > 0).mean())})
    out[k] = r
json.dump(out, open("results/strict_tags.json", "w"), indent=2)
print(json.dumps(out["_counts"]), "\n", out["_top_titles"])
for k in ["cap10", "pen0.4", "cap3", "ca_slot"]:
    print(k, {a: (round(b, 4) if b is not None else None) for a, b in out[k].items() if a.startswith("CA")})
