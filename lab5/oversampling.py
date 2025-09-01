import numpy as np
import pandas as pd
from collections import Counter


def get_split_indices(n, train_size, val_size, test_size, random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_train = int(train_size * n)
    n_val = int(val_size * n)
    if n_train == 0 and n > 0:
        n_train = 1
    if n_val == 0 and n - n_train > 1:
        n_val = 1
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


def train_linear_regression(X_train, y_train, X_val, y_val, lr=0.01, epochs=200, patience=10):
    n_train, d = X_train.shape
    X_tr = np.hstack([np.ones((n_train, 1)), X_train])
    n_val = X_val.shape[0]
    X_v = np.hstack([np.ones((n_val, 1)), X_val])
    w = np.zeros(d + 1, dtype=float)
    best_w = w.copy()
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(1, epochs + 1):
        preds = X_tr.dot(w)
        error = preds - y_train
        loss = (error ** 2).mean()
        grad = (2.0 / n_train) * X_tr.T.dot(error)
        w -= lr * grad

        val_preds = X_v.dot(w)
        val_error = val_preds - y_val
        val_loss = (val_error ** 2).mean()

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_w = w.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_w, best_val_loss


def predict_linear_regression(w, X):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_aug.dot(w)


def binary_metrics(y_true, y_pred_bin):
    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {'acc': accuracy, 'prec': precision, 'rec': recall, 'f1': f1}


def random_oversample(X, y, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    counter = Counter(y)
    if len(counter) <= 1:
        return X.copy(), y.copy()
    # target is majority count
    target = max(counter.values())
    X_res = []
    y_res = []
    for cls in sorted(counter.keys()):
        idx = np.where(y == cls)[0]
        n = len(idx)
        if n == 0:
            continue
        reps = target // n
        rem = target - reps * n
        chosen = np.concatenate([np.repeat(idx, reps), rng.choice(idx, rem, replace=True)]) if reps > 0 or rem > 0 else idx
        X_res.append(X[chosen])
        y_res.append(y[chosen])
    X_out = np.vstack(X_res)
    y_out = np.concatenate(y_res)
    # shuffle
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

            methods = ['none', 'random']
            results = {}

            for m in methods:
                if m == 'none':
                    X_tr_used, y_tr_used = X_train_s, y_train
                elif m == 'random':
                    X_tr_used, y_tr_used = random_oversample(X_train_s, y_train, rng=rng)

                # For validation we keep original val set (no oversampling)
                w, vloss = train_linear_regression(X_tr_used, y_tr_used, X_val_s, y_val, lr=0.05, epochs=300, patience=15)
                preds = predict_linear_regression(w, X_test_s)
                preds_bin = (preds > 0.5).astype(int)
                metrics = binary_metrics(y_test, preds_bin)
                results[m] = metrics

            print(f"Class: {name}")
            for m in methods:
                met = results[m]
                print(f"  {m:6} acc={met['acc']:.3f} prec={met['prec']:.3f} rec={met['rec']:.3f} f1={met['f1']:.3f}")


if __name__ == '__main__':
    run_experiment()
