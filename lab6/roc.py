import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lab5.lab_utils import get_split_indices, standardize_fit, standardize_apply, predict_linear_regression, train_linear_regression


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_roc(y_true_bin, scores, thresholds):
    # thresholds assumed in [0,1]; scores must be in [0,1]
    tpr_list = []
    fpr_list = []
    P = int((y_true_bin == 1).sum())
    N = int((y_true_bin == 0).sum())
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_true_bin == 1) & (y_pred == 1)).sum())
        fp = int(((y_true_bin == 0) & (y_pred == 1)).sum())
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return np.array(fpr_list), np.array(tpr_list)


def auc_from_curve(fpr, tpr):
    # ensure sorted by fpr ascending
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]
    return float(np.trapz(t, f))


def run():
    # load data
    df = pd.read_csv('lab5/Iris.csv')
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy(dtype=float)
    y_str = df['Species'].to_numpy()
    classes = np.unique(y_str)
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_int[s] for s in y_str])

    # split
    train_idx, val_idx, test_idx = get_split_indices(X.shape[0], 0.8, 0.1, 0.1, random_state=42)
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # standardize
    mu, sigma = standardize_fit(X_train)
    X_train_s = standardize_apply(X_train, mu, sigma)
    X_val_s = standardize_apply(X_val, mu, sigma)
    X_test_s = standardize_apply(X_test, mu, sigma)

    # train one-vs-rest linear regressors
    n_classes = len(classes)
    Ws = []
    for cls_idx in range(n_classes):
        ytr_bin = (y_train == cls_idx).astype(float)
        yval_bin = (y_val == cls_idx).astype(float)
        w, *_ = train_linear_regression(X_train_s, ytr_bin, X_val_s, yval_bin, lr=0.05, epochs=1000, patience=30)
        Ws.append(w)

    # get continuous scores and map to [0,1] via sigmoid
    scores_raw = np.vstack([predict_linear_regression(w, X_test_s) for w in Ws]).T
    scores_sig = sigmoid(scores_raw)

    thresholds = np.linspace(0.0, 1.0, 101)

    # compute ROC and AUC per class
    results = {}
    for i in range(n_classes):
        y_true_bin = (y_test == i).astype(int)
        scores_i = scores_sig[:, i]
        fpr, tpr = compute_roc(y_true_bin, scores_i, thresholds)
        auc = auc_from_curve(fpr, tpr)
        results[i] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}

    # print AUCs
    print('Per-class AUC (using sigmoid(scores) and thresholds 0..1):')
    for i, cls in enumerate(classes):
        print(f" Class {i} ({cls}): AUC = {results[i]['auc']:.4f}")

    # optional: try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(n_classes):
            plt.plot(results[i]['fpr'], results[i]['tpr'], label=f"{i} {classes[i]} (AUC={results[i]['auc']:.3f})")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves (one-vs-rest)')
        plt.legend()
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            # if no GUI, save to file
            plt.savefig('lab6/roc_curves.png')
            print('Saved ROC plot to lab6/roc_curves.png')
    except Exception:
        print('matplotlib not available; skipping plot')


if __name__ == '__main__':
    run()
