import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DBSCAN:
    """Minimal DBSCAN from scratch."""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def _dist(self, a, b):
        return np.linalg.norm(a - b)

    def _neighbors(self, X, i):
        d = np.linalg.norm(X - X[i], axis=1)
        return np.where(d <= self.eps)[0].tolist()

    def fit_predict(self, X):
        n = len(X)
        labels = -1 * np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cid = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neigh = self._neighbors(X, i)
            if len(neigh) < self.min_samples:
                continue
            labels[i] = cid
            seeds = list(neigh)
            j = 0
            while j < len(seeds):
                p = seeds[j]
                if not visited[p]:
                    visited[p] = True
                    neigh_p = self._neighbors(X, p)
                    if len(neigh_p) >= self.min_samples:
                        for q in neigh_p:
                            if q not in seeds:
                                seeds.append(q)
                if labels[p] == -1:
                    labels[p] = cid
                j += 1
            cid += 1
        return labels


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / (sigma + 1e-12)


def silhouette(X, labels):
    # compute silhouette for non-noise points
    mask = labels != -1
    if mask.sum() <= 1:
        return 0.0
    Xn = X[mask]
    L = labels[mask]
    s_vals = []
    for idx, x in enumerate(Xn):
        same = Xn[L == L[idx]]
        a = np.mean(np.linalg.norm(same - x, axis=1)) if len(same) > 1 else 0.0
        b = np.inf
        for lab in np.unique(L):
            if lab == L[idx]:
                continue
            other = Xn[L == lab]
            if len(other):
                b = min(b, np.mean(np.linalg.norm(other - x, axis=1)))
        s = 0.0 if b == np.inf else (b - a) / max(a, b)
        s_vals.append(s)
    return float(np.mean(s_vals)) if s_vals else 0.0


def adjusted_rand_index(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)
    labels_true = np.unique(true)
    labels_pred = np.unique(pred)
    n = len(true)
    # contingency
    cont = np.zeros((labels_true.size, labels_pred.size), dtype=int)
    for i, t in enumerate(labels_true):
        for j, p in enumerate(labels_pred):
            cont[i, j] = np.sum((true == t) & (pred == p))
    sum_comb = lambda a: np.sum(a * (a - 1) // 2)
    A = sum_comb(cont.sum(axis=1))
    B = sum_comb(cont.sum(axis=0))
    C = sum_comb(cont.flatten())
    denom = 0.5 * (A + B) - (A * B) / (n * (n - 1) / 2)
    if denom == 0:
        return 1.0
    return float((C - (A * B) / (n * (n - 1) / 2)) / denom)


def nmi(true, pred):
    # normalized mutual information (simple)
    true = np.array(true)
    pred = np.array(pred)
    n = len(true)
    mi = 0.0
    for t in np.unique(true):
        for p in np.unique(pred):
            nij = np.sum((true == t) & (pred == p))
            if nij:
                ni = np.sum(true == t)
                nj = np.sum(pred == p)
                mi += (nij / n) * np.log((n * nij) / (ni * nj) + 1e-12)
    h = lambda arr: -np.sum([(c / n) * np.log(c / n + 1e-12) for c in np.bincount(arr)])
    h_true = h(np.searchsorted(np.unique(true), true))
    h_pred = h(np.searchsorted(np.unique(pred), pred))
    return float(2 * mi / (h_true + h_pred + 1e-12))


def load_iris(local_path=None):
    if local_path and os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # normalize column names used in previous assignment
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
        df = df.rename(columns={
            'SepalLengthCm': 'sepal_length',
            'SepalWidthCm': 'sepal_width',
            'PetalLengthCm': 'petal_length',
            'PetalWidthCm': 'petal_width',
            'Species': 'species'
        })
    else:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    return df


def main():
    local = r'D:\U23AI118\SEM 5\ML-Lab\lab5\Iris.csv'
    df = load_iris(local)
    X = df.iloc[:, :-1].values.astype(float)
    y = df['species'].values
    Xs = standardize(X)

    model = DBSCAN(eps=0.5, min_samples=5)
    pred = model.fit_predict(Xs)

    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    n_noise = list(pred).count(-1)
    print(f"clusters: {n_clusters}, noise: {n_noise}")
    print(f"silhouette: {silhouette(Xs, pred):.4f}")
    print(f"ARI: {adjusted_rand_index(y, pred):.4f}")
    print(f"NMI: {nmi(y, pred):.4f}")

    # simple 2x2 plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(X[:, 2], X[:, 3], c=pred, cmap='viridis', s=30)
    ax[0].set_title('DBSCAN clusters (petal)')
    for i, lab in enumerate(np.unique(y)):
        mask = y == lab
        ax[1].scatter(X[mask, 2], X[mask, 3], label=lab, s=30)
    ax[1].set_title('True labels (petal)')
    ax[1].legend(fontsize='small')
    plt.tight_layout()
    plt.savefig('dbscan_iris_compact.png', dpi=200)
    print('saved dbscan_iris_compact.png')


if __name__ == '__main__':
    main()
