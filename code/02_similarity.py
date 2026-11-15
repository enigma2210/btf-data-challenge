import sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

P = sys.argv[1] if len(sys.argv) > 1 else "data/prepared"
K = 40
t0 = time.time()
train = pd.read_pickle(f"{P}/train.pkl")
eligible = np.load(f"{P}/eligible.npy")
col = pd.Series(np.arange(len(eligible)), index=eligible)
train = train[train.movieId.isin(col.index)]
uids, urow = np.unique(train.userId.values, return_inverse=True)
X = sp.csr_matrix((np.ones(len(train), np.float32), (urow, col[train.movieId].values)),
                  shape=(len(uids), len(eligible)))
X.data[:] = 1.0
norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel()).astype(np.float32)
Xn = (X @ sp.diags(1.0 / norms)).tocsc()
XT = Xn.T.tocsr()
n = len(eligible); B = 800
rows, cols, vals = [], [], []
for s in range(0, n, B):
    e = min(n, s + B)
    blk = (XT @ Xn[:, s:e]).toarray()
    for j in range(e - s):
        blk[s + j, j] = 0.0
    top = np.argpartition(-blk, K, axis=0)[:K]
    for j in range(e - s):
        nb = top[:, j]; v = blk[nb, j]
        keep = v > 0
        rows.append(np.full(keep.sum(), s + j)); cols.append(nb[keep]); vals.append(v[keep])
    print(f"{e}/{n} items  {time.time()-t0:.0f}s", flush=True)
S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
sp.save_npz(f"{P}/sim_top40.npz", S)
print("done", S.nnz, f"{time.time()-t0:.0f}s")
