import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lab5.lab_utils import get_split_indices, standardize_fit, standardize_apply


def normal_equation(X, y, l2_reg=0.0):
    # X already augmented with bias column
    n_features = X.shape[1]
    A = X.T.dot(X)
    if l2_reg > 0:
        A = A + l2_reg * np.eye(n_features)
    # use pseudo-inverse for stability
    w = np.linalg.pinv(A).dot(X.T).dot(y)
    return w


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def run():
    df = pd.read_csv('lab5/Iris.csv')
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy(dtype=float)
    y_str = df['Species'].to_numpy()
    classes = np.unique(y_str)
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_int[s] for s in y_str], dtype=float)

    # split
    train_idx, val_idx, test_idx = get_split_indices(X.shape[0], 0.8, 0.1, 0.1, random_state=42)
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # standardize using train
    mu, sigma = standardize_fit(X_train)
    X_train_s = standardize_apply(X_train, mu, sigma)
    X_val_s = standardize_apply(X_val, mu, sigma)
    X_test_s = standardize_apply(X_test, mu, sigma)

    # augment bias
    X_tr_aug = np.hstack([np.ones((X_train_s.shape[0], 1)), X_train_s])
    X_test_aug = np.hstack([np.ones((X_test_s.shape[0], 1)), X_test_s])

    # fit via normal equation (no sklearn)
    w = normal_equation(X_tr_aug, y_train, l2_reg=0.0)

    # predictions on test set
    y_pred_test = X_test_aug.dot(w)

    # compute errors
    mae_val = mae(y_test, y_pred_test)
    mse_val = mse(y_test, y_pred_test)
    r2_val = r2_score(y_test, y_pred_test)

    print('Linear regression via normal equation predicting integer-encoded species (0..2)')
    print(f'Weights (bias first): {w}')
    print('\nTest set results:')
    for i in range(len(y_test)):
        print(f'  true={int(y_test[i])}  pred={y_pred_test[i]:.4f}')

    print('\nError metrics on test set:')
    print(f'  MAE = {mae_val:.4f}')
    print(f'  MSE = {mse_val:.4f}')
    print(f'  R2  = {r2_val:.4f}')


if __name__ == '__main__':
    run()
