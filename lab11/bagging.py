"""Bagging ensemble implemented from scratch for the Iris dataset.

This script loads `lab5/Iris.csv`, encodes labels, splits into train/val/test
sets, trains a Bagging ensemble of decision stumps (single-feature threshold
learners) and evaluates on validation and test sets.

No external ML libraries are used.
"""
from collections import Counter, defaultdict
import csv
import math
import os
import sys
import random
from typing import Tuple, List

import numpy as np

# Try to reuse existing helper if available in lab5
try:
    from ..lab5 import lab_utils as lu
except Exception:
    # allow running directly when script executed from lab11 folder
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from lab5 import lab_utils as lu
    except Exception:
        # minimal fallback implementations used below if needed
        lu = None


def load_iris_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Iris CSV and return features X (n x d), labels y (n,), and class names.
    Expects file to have header with Species as last column.
    """
    xs = []
    ys = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip Id column if present
            feats = [float(row[k]) for k in reader.fieldnames if k not in ('Id', 'Species')]
            xs.append(feats)
            ys.append(row['Species'])
    X = np.array(xs, dtype=float)
    y_raw = np.array(ys)
    classes = sorted(list(set(y_raw)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_raw], dtype=int)
    return X, y, classes


class DecisionStump:
    """Decision stump for multiclass classification.

    Learns a single feature and threshold that partitions data into two leaves.
    Prediction is majority class in the leaf.
    """

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_label = None
        self.right_label = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        best_score = float('inf')
        best_feat = 0
        best_thr = 0.0
        best_left_label = None
        best_right_label = None

        for feat in range(d):
            vals = X[:, feat]
            # sort indices by feature value
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_y = y[order]
            # candidate thresholds are midpoints between consecutive unique values
            for i in range(1, n):
                if sorted_vals[i] == sorted_vals[i - 1]:
                    continue
                thr = 0.5 * (sorted_vals[i] + sorted_vals[i - 1])
                left_y = sorted_y[:i]
                right_y = sorted_y[i:]
                # compute weighted gini
                def gini(arr):
                    if arr.size == 0:
                        return 0.0
                    counts = np.bincount(arr, minlength=int(y.max()) + 1)
                    probs = counts / counts.sum()
                    return 1.0 - np.sum(probs ** 2)

                g_left = gini(left_y)
                g_right = gini(right_y)
                score = (i * g_left + (n - i) * g_right) / n
                if score < best_score:
                    best_score = score
                    best_feat = feat
                    best_thr = thr
                    # majority labels
                    best_left_label = Counter(left_y).most_common(1)[0][0] if left_y.size > 0 else 0
                    best_right_label = Counter(right_y).most_common(1)[0][0] if right_y.size > 0 else 0

        self.feature_index = int(best_feat)
        self.threshold = float(best_thr)
        self.left_label = int(best_left_label)
        self.right_label = int(best_right_label)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature_index is None:
            raise RuntimeError('Stump not fitted')
        vals = X[:, self.feature_index]
        out = np.where(vals <= self.threshold, self.left_label, self.right_label)
        return out


class BaggingClassifier:
    """Bagging ensemble of base estimators (DecisionStump by default).

    Parameters
    ----------
    n_estimators: int
        Number of bootstrap estimators.
    max_features: int or None
        Not used for stump; included for API compatibility.
    rng: random.Random or None
        Random generator.
    """

    def __init__(self, n_estimators: int = 25, rng: random.Random = None):
        self.n_estimators = int(n_estimators)
        self.rng = rng or random.Random(0)
        self.estimators_: List[DecisionStump] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        self.estimators_ = []
        for i in range(self.n_estimators):
            # bootstrap sample indices
            idx = [self.rng.randrange(n) for _ in range(n)]
            Xb = X[idx]
            yb = y[idx]
            stump = DecisionStump()
            stump.fit(Xb, yb)
            self.estimators_.append(stump)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError('Ensemble not fitted')
        # collect predictions
        preds = np.vstack([est.predict(X) for est in self.estimators_])
        # majority vote along axis 0
        out = []
        for col in preds.T:
            counts = Counter(col)
            # tie-breaker: choose smallest label
            most_common = counts.most_common()
            most_common.sort(key=lambda x: (-x[1], x[0]))
            out.append(most_common[0][0])
        return np.array(out, dtype=int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    script_dir = os.path.dirname(__file__)
    iris_path = os.path.join(script_dir, '..', 'lab5', 'Iris.csv')
    iris_path = os.path.normpath(iris_path)
    if not os.path.exists(iris_path):
        print('Could not find Iris.csv at', iris_path)
        return

    X, y, classes = load_iris_csv(iris_path)
    n = X.shape[0]
    # use helper split if available
    if lu is not None:
        train_idx, val_idx, test_idx = lu.get_split_indices(n, 0.6, 0.2, 0.2, random_state=123)
    else:
        rng = np.random.RandomState(123)
        perm = rng.permutation(n)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print('dataset sizes: train=%d val=%d test=%d' % (len(y_train), len(y_val), len(y_test)))

    # Train bagging ensemble
    bag = BaggingClassifier(n_estimators=25, rng=random.Random(0))
    bag.fit(X_train, y_train)

    y_val_pred = bag.predict(X_val)
    y_test_pred = bag.predict(X_test)

    val_acc = accuracy(y_val, y_val_pred)
    test_acc = accuracy(y_test, y_test_pred)

    print('\nValidation accuracy: %.4f' % val_acc)
    print('Test accuracy:       %.4f' % test_acc)

    cm_val = confusion_matrix(y_val, y_val_pred, n_classes=len(classes))
    cm_test = confusion_matrix(y_test, y_test_pred, n_classes=len(classes))

    print('\nValidation confusion matrix (rows=true, cols=pred):')
    print(cm_val)
    print('\nTest confusion matrix (rows=true, cols=pred):')
    print(cm_test)

    # Optionally save test predictions
    out_csv = os.path.join(script_dir, 'test_predictions_bagging.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['TrueLabel', 'PredLabel'])
        for t, p in zip(y_test, y_test_pred):
            writer.writerow([classes[int(t)], classes[int(p)]])
    print('\nWrote test predictions to', out_csv)


if __name__ == '__main__':
    main()
