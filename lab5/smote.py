import numpy as np
import pandas as pd
from collections import Counter
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lab5.lab_utils import get_split_indices, standardize_fit, standardize_apply, random_oversample, binary_metrics, predict_linear_regression, train_linear_regression

def smote_pairwise(X_minority, n_samples, rng=None):
    """Generate synthetic samples by randomly picking two minority samples and
    interpolating between them (uniform gap).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    n_min = X_minority.shape[0]
    if n_min == 0:
        return np.empty((0, X_minority.shape[1]))
    synth = []
    for _ in range(n_samples):
        i, j = rng.choice(n_min, size=2, replace=False)
        gap = rng.rand()
        synth.append(X_minority[i] + gap * (X_minority[j] - X_minority[i]))
    return np.vstack(synth)


def smote_nearest(X_minority, n_samples, rng=None):
    """Generate synthetic samples by interpolating between each sample and its
    nearest neighbor (within minority class).
    For generating n_samples we randomly pick base points and use their nearest neighbor.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    n_min = X_minority.shape[0]
    if n_min == 0:
        return np.empty((0, X_minority.shape[1]))
    if n_min == 1:
        # jitter the single sample
        jitter = rng.normal(scale=1e-3, size=(n_samples, X_minority.shape[1]))
        return np.repeat(X_minority, n_samples, axis=0) + jitter

    # compute pairwise distances and nearest neighbor indices
    dists = np.linalg.norm(X_minority[:, None, :] - X_minority[None, :, :], axis=2)
    # set diag to large
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argmin(dists, axis=1)

    synth = []
    for _ in range(n_samples):
        i = rng.randint(0, n_min)
        j = nn_idx[i]
        gap = rng.rand()
        synth.append(X_minority[i] + gap * (X_minority[j] - X_minority[i]))
    return np.vstack(synth)


def oversample_smote_pair(X, y, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    counter = Counter(y)
    if len(counter) <= 1:
        return X.copy(), y.copy()
    target = max(counter.values())
    X_res = []
    y_res = []
    for cls in sorted(counter.keys()):
        idx = np.where(y == cls)[0]
        n = len(idx)
        if n == 0:
            continue
        if n < target:
            need = target - n
            X_min = X[idx]
            synth = smote_pairwise(X_min, need, rng=rng)
            X_cat = np.vstack([X_min, synth])
            y_cat = np.array([cls] * X_cat.shape[0])
        else:
            X_cat = X[idx]
            y_cat = np.array([cls] * n)
        X_res.append(X_cat)
        y_res.append(y_cat)
    X_out = np.vstack(X_res)
    y_out = np.concatenate(y_res)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def oversample_smote_nearest(X, y, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    counter = Counter(y)
    if len(counter) <= 1:
        return X.copy(), y.copy()
    target = max(counter.values())
    X_res = []
    y_res = []
    for cls in sorted(counter.keys()):
        idx = np.where(y == cls)[0]
        n = len(idx)
        if n == 0:
            continue
        if n < target:
            need = target - n
            X_min = X[idx]
            synth = smote_nearest(X_min, need, rng=rng)
            X_cat = np.vstack([X_min, synth])
            y_cat = np.array([cls] * X_cat.shape[0])
        else:
            X_cat = X[idx]
            y_cat = np.array([cls] * n)
        X_res.append(X_cat)
        y_res.append(y_cat)
    X_out = np.vstack(X_res)
    y_out = np.concatenate(y_res)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def run_experiment():
    df = pd.read_csv('lab5/Iris.csv')
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy(dtype=float)
    labels = {
        'setosa': (df['Species'] == 'Iris-setosa').astype(int).to_numpy(),
        'versicolor': (df['Species'] == 'Iris-versicolor').astype(int).to_numpy(),
        'virginica': (df['Species'] == 'Iris-virginica').astype(int).to_numpy(),
    }

    splits = [(0.8, 0.1, 0.1), (0.7, 0.15, 0.15)]
    rng = np.random.RandomState(42)

    for train_size, val_size, test_size in splits:
        n = X.shape[0]
        train_idx, val_idx, test_idx = get_split_indices(n, train_size, val_size, test_size, random_state=42)
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        mu, sigma = standardize_fit(X_train)
        X_train_s = standardize_apply(X_train, mu, sigma)
        X_val_s = standardize_apply(X_val, mu, sigma)
        X_test_s = standardize_apply(X_test, mu, sigma)

        print('\nSplit {}:{}:{}'.format(int(train_size*100), int(val_size*100), int(test_size*100)))

        for name, y_all in labels.items():
            y_train = y_all[train_idx]
            y_val = y_all[val_idx]
            y_test = y_all[test_idx]

            methods = ['none', 'random_dup', 'smote_pair', 'smote_nearest']
            results = {}

            for m in methods:
                if m == 'none':
                    X_tr_used, y_tr_used = X_train_s, y_train
                elif m == 'random_dup':
                    X_tr_used, y_tr_used = random_oversample(X_train_s, y_train, rng=rng)
                elif m == 'smote_pair':
                    X_tr_used, y_tr_used = oversample_smote_pair(X_train_s, y_train, rng=rng)
                else:
                    X_tr_used, y_tr_used = oversample_smote_nearest(X_train_s, y_train, rng=rng)

                w, vloss, *_ = train_linear_regression(X_tr_used, y_tr_used, X_val_s, y_val, lr=0.05, epochs=300, patience=15)
                preds = predict_linear_regression(w, X_test_s)
                preds_bin = (preds > 0.5).astype(int)
                metrics = binary_metrics(y_test, preds_bin)
                results[m] = metrics

            print(f"Class: {name}")
            for m in methods:
                met = results[m]
                print(f"  {m:12} acc={met['acc']:.3f} prec={met['prec']:.3f} rec={met['rec']:.3f} f1={met['f1']:.3f}")


if __name__ == '__main__':
    run_experiment()
