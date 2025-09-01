import numpy as np
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


def predict_linear_regression(w, X):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_aug.dot(w)


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


def train_linear_regression(X_train, y_train, X_val, y_val, lr=0.01, epochs=200, patience=10):
    """Train linear regression by gradient descent with early stopping on validation MSE.
    Returns best_w, best_val_loss, epoch_ran, train_losses, val_losses for diagnostics.
    """
    n_train, d = X_train.shape
    X_tr = np.hstack([np.ones((n_train, 1)), X_train])
    n_val = X_val.shape[0]
    X_v = np.hstack([np.ones((n_val, 1)), X_val])
    w = np.zeros(d + 1, dtype=float)

    best_w = w.copy()
    best_val_loss = float('inf')
    wait = 0
    train_losses = []
    val_losses = []
    epoch_ran = 0

    for epoch in range(1, epochs + 1):
        preds = X_tr.dot(w)
        error = preds - y_train
        loss = (error ** 2).mean()
        train_losses.append(loss)

        grad = (2.0 / n_train) * X_tr.T.dot(error)
        w -= lr * grad

        val_preds = X_v.dot(w)
        val_error = val_preds - y_val
        val_loss = (val_error ** 2).mean()
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_w = w.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                epoch_ran = epoch
                break

    if epoch_ran == 0:
        epoch_ran = epoch

    return best_w, best_val_loss, epoch_ran, train_losses, val_losses

