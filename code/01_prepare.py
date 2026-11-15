import sys, os, json
import numpy as np, pandas as pd

ML = sys.argv[1] if len(sys.argv) > 1 else "data/ml-32m"
WD = sys.argv[2] if len(sys.argv) > 2 else "data/wikidata_ca_fr.csv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/prepared"
os.makedirs(OUT, exist_ok=True)

r = pd.read_csv(f"{ML}/ratings.csv", dtype={"userId": "int32", "movieId": "int32",
                                            "rating": "float32", "timestamp": "int64"})
stats = {"ratings": int(len(r))}
pos = r[r.rating >= 4.0][["userId", "movieId", "timestamp"]]
del r
stats["positives"] = int(len(pos))

pos = pos.sort_values(["userId", "timestamp", "movieId"], kind="mergesort")
cnt = pos.groupby("userId").movieId.transform("size")
rank_from_end = pos.groupby("userId").cumcount(ascending=False)
is_hold = (cnt >= 15) & (rank_from_end < 5)
stats["users_ge15"] = int(pos.loc[cnt >= 15, "userId"].nunique())

hold = pos[is_hold][["userId", "movieId"]]
train = pos[~is_hold][["userId", "movieId"]]
item_train = train.movieId.value_counts()
eligible = np.sort(item_train[item_train >= 40].index.values)
stats["eligible_items"] = int(len(eligible))
stats["train_positives"] = int(len(train))

train.to_pickle(f"{OUT}/train.pkl")
hold.to_pickle(f"{OUT}/hold.pkl")
np.save(f"{OUT}/eligible.npy", eligible)
item_train.rename("train_pos").to_frame().to_pickle(f"{OUT}/item_train_pos.pkl")

mv = pd.read_csv(f"{ML}/movies.csv")
mv["year"] = pd.to_numeric(mv.title.str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
ln = pd.read_csv(f"{ML}/links.csv", dtype={"imdbId": str})
ln["imdb"] = "tt" + ln.imdbId.str.zfill(7)
wd = pd.read_csv(WD)
stats["wikidata_rows"] = int(len(wd))
stats["wikidata_rows_by_tag"] = wd.tag.value_counts().to_dict()
wd_ca = set(wd.loc[wd.tag == "CA", "imdb"]); wd_fr = set(wd.loc[wd.tag == "FR", "imdb"])

norm = lambda s: "tt" + s[2:].lstrip("0")
wd_ca = {norm(x) for x in wd_ca}; wd_fr = {norm(x) for x in wd_fr}
ln["imdb_n"] = ln.imdb.map(norm)
ln["is_ca"] = ln.imdb_n.isin(wd_ca); ln["is_fr"] = ln.imdb_n.isin(wd_fr)
mv = mv.merge(ln[["movieId", "imdb", "is_ca", "is_fr"]], on="movieId", how="left")
mv[["is_ca", "is_fr"]] = mv[["is_ca", "is_fr"]].fillna(False)
mv["train_pos"] = mv.movieId.map(item_train).fillna(0).astype(int)
mv["eligible"] = mv.movieId.isin(eligible)
mv.to_pickle(f"{OUT}/movies.pkl")

el = mv[mv.eligible]
stats["catalog_movies"] = int(len(mv))
stats["catalog_ca"] = int(mv.is_ca.sum()); stats["catalog_fr"] = int(mv.is_fr.sum())
stats["eligible_ca"] = int(el.is_ca.sum()); stats["eligible_fr"] = int(el.is_fr.sum())
stats["eligible_ca_and_fr"] = int((el.is_ca & el.is_fr).sum())
json.dump(stats, open(f"{OUT}/prepare_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
