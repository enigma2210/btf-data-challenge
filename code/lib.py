import numpy as np, pandas as pd, scipy.sparse as sp

N_LIST = 10

class Data:
    def __init__(self, P="data/prepared", n_users=12000, seed=2026, exclude=None):
        self.P = P
        self.eligible = np.load(f"{P}/eligible.npy")
        self.col = pd.Series(np.arange(len(self.eligible)), index=self.eligible)
        mv = pd.read_pickle(f"{P}/movies.pkl").set_index("movieId").loc[self.eligible]
        self.mv = mv
        n = len(self.eligible)
        self.pop = mv.train_pos.values.astype(np.int64)

        order = self.pop.argsort(kind="mergesort")
        pct = np.empty(n); pct[order] = (np.arange(n) + 0.5) / n * 100
        self.pop_pct = pct
        self.head_rank = np.argsort(-self.pop, kind="mergesort")
        self.year = mv.year.values
        self.genres = [set(g.split("|")) - {"(no genres listed)"} for g in mv.genres.values]
        allg = sorted(set().union(*self.genres))
        self.genre_names = allg
        G = np.zeros((n, len(allg)), np.float32)
        gi = {g: i for i, g in enumerate(allg)}
        for k, gs in enumerate(self.genres):
            for g in gs: G[k, gi[g]] = 1
        self.G = G
        self.tags = {"CA": mv.is_ca.values.astype(bool), "FR": mv.is_fr.values.astype(bool)}
        self.S = sp.load_npz(f"{P}/sim_top40.npz").tocsr()

        hold = pd.read_pickle(f"{P}/hold.pkl")
        users = np.sort(hold.userId.unique())
        if exclude is not None:
            users = np.setdiff1d(users, exclude)
        rng = np.random.default_rng(seed)
        self.users = np.sort(rng.choice(users, size=n_users, replace=False))
        uidx = pd.Series(np.arange(n_users), index=self.users)
        train = pd.read_pickle(f"{P}/train.pkl")
        train = train[train.userId.isin(self.users) & train.movieId.isin(self.col.index)]
        self.U = sp.csr_matrix((np.ones(len(train), np.float32),
                                (uidx[train.userId].values, self.col[train.movieId].values)),
                               shape=(n_users, n))
        hold = hold[hold.userId.isin(self.users)]
        self.hold_all = hold.groupby("userId").size().reindex(self.users).values
        h = hold[hold.movieId.isin(self.col.index)]
        self.H = sp.csr_matrix((np.ones(len(h), np.float32),
                                (uidx[h.userId].values, self.col[h.movieId].values)),
                               shape=(n_users, n))

        gcount = self.U @ self.G
        self.dom_genre = np.asarray(gcount.argmax(axis=1)).ravel()

    def personalized_scores(self, rows):
        sc = (self.U[rows] @ self.S).toarray()
        sc[self.U[rows].toarray() > 0] = -np.inf
        return sc

    def popularity_scores(self, rows):
        sc = np.tile(self.pop.astype(np.float64), (len(rows), 1))
        sc[self.U[rows].toarray() > 0] = -np.inf
        return sc

    def random_scores(self, rows, rng):
        sc = rng.random((len(rows), len(self.eligible)))
        sc[self.U[rows].toarray() > 0] = -np.inf
        return sc

def topn(scores, n=N_LIST):
    idx = np.argpartition(-scores, n, axis=1)[:, :n]
    o = np.take_along_axis(scores, idx, 1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, o, 1)

def rerank_cap(scores, mask, cap, n=N_LIST, depth=200):
    idx = topn(scores, depth)
    out = np.empty((scores.shape[0], n), np.int64)
    for u in range(scores.shape[0]):
        m = mask if mask.ndim == 1 else mask[u]
        used = 0; k = 0
        for it in idx[u]:
            if m[it]:
                if used >= cap: continue
                used += 1
            out[u, k] = it; k += 1
            if k == n: break
        if k < n:
            rest = [it for it in idx[u] if it not in out[u, :k]][: n - k]
            out[u, k:] = rest
    return out

def gini(x):
    x = np.sort(np.asarray(x, np.float64))
    n = len(x)
    if x.sum() == 0: return 0.0
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())

def label(D, lists, hits_by_user=None):
    flat = lists.ravel()
    n = len(D.eligible)
    exp = np.bincount(flat, minlength=n)
    head1 = np.zeros(n, bool); head1[D.head_rank[: int(round(n * 0.01))]] = True
    H = D.H
    hits = np.array([H[u, lists[u]].sum() for u in range(lists.shape[0])])
    n_hold_el = np.asarray(H.sum(axis=1)).ravel()

    G = D.G[lists]
    inter = np.einsum("uig,ujg->uij", G, G)
    sz = G.sum(-1)
    union = sz[:, :, None] + sz[:, None, :] - inter
    jd = 1 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    iu = np.triu_indices(N_LIST, 1)
    spread = jd[:, iu[0], iu[1]].mean()
    out = {
        "recall@10": float(hits.sum() / (5 * len(hits))),
        "recall@10_eligible_holdout": float(hits.sum() / max(n_hold_el.sum(), 1)),
        "pop_pct_mean": float(D.pop_pct[flat].mean()),
        "head1_share": float(head1[flat].mean()),
        "shelf_used": int((exp > 0).sum()),
        "shelf_used_share": float((exp > 0).mean()),
        "slot_gini": float(gini(exp)),
        "share_1990s": float(((D.year[flat] >= 1990) & (D.year[flat] < 2000)).mean()),
        "genre_spread": float(spread),
    }
    for t, m in D.tags.items():
        cat = m.mean()
        sl = m[flat].mean()
        out[f"{t}_catalog_share"] = float(cat)
        out[f"{t}_slot_share"] = float(sl)
        out[f"{t}_exposure_ratio"] = float(sl / cat) if cat > 0 else None
        out[f"{t}_screens_with_any"] = float((m[lists].sum(1) > 0).mean())
    return out, hits
