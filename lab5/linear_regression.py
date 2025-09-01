import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lab5.lab_utils import get_split_indices, standardize_fit, standardize_apply, train_linear_regression, predict_linear_regression

# Load data
df = pd.read_csv('lab5/Iris.csv')

# Features and class labels
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy(dtype=float)
y1 = (df['Species'] == 'Iris-setosa').astype(int).to_numpy()
y2 = (df['Species'] == 'Iris-versicolor').astype(int).to_numpy()
y3 = (df['Species'] == 'Iris-virginica').astype(int).to_numpy()


# use train_linear_regression and predict_linear_regression from lab_utils


def accuracy(y_true, y_pred_binary):
    return float((y_true == y_pred_binary).mean())


splits = [(0.8, 0.1, 0.1), (0.7, 0.15, 0.15)]
for train_size, val_size, test_size in splits:
    # create indices once and reuse for all labels so validation is independent and consistent
    n = X.shape[0]
    train_idx, val_idx, test_idx = get_split_indices(n, train_size, val_size, test_size, random_state=42)
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    y1_train = y1[train_idx]
    y1_val = y1[val_idx]
    y1_test = y1[test_idx]
    y2_train = y2[train_idx]
    y2_val = y2[val_idx]
    y2_test = y2[test_idx]
    y3_train = y3[train_idx]
    y3_val = y3[val_idx]
    y3_test = y3[test_idx]

    # Standardize using training set statistics
    mu, sigma = standardize_fit(X_train)
    X_train_s = standardize_apply(X_train, mu, sigma)
    X_val_s = standardize_apply(X_val, mu, sigma)
    X_test_s = standardize_apply(X_test, mu, sigma)

    # Training hyperparameters
    lr = 0.05
    epochs = 300
    patience = 15

    # Train three one-vs-rest linear regression models with early stopping on validation loss
    w1, vloss1, e1, tloss1, vloss_hist1 = train_linear_regression(X_train_s, y1_train, X_val_s, y1_val, lr=lr, epochs=epochs, patience=patience)
    w2, vloss2, e2, tloss2, vloss_hist2 = train_linear_regression(X_train_s, y2_train, X_val_s, y2_val, lr=lr, epochs=epochs, patience=patience)
    w3, vloss3, e3, tloss3, vloss_hist3 = train_linear_regression(X_train_s, y3_train, X_val_s, y3_val, lr=lr, epochs=epochs, patience=patience)

    # Predict on test set and threshold at 0.5
    y1_pred = (predict_linear_regression(w1, X_test_s) > 0.5).astype(int)
    y2_pred = (predict_linear_regression(w2, X_test_s) > 0.5).astype(int)
    y3_pred = (predict_linear_regression(w3, X_test_s) > 0.5).astype(int)

    # Evaluate
    print(f"Split {int(train_size*100)}:{int(val_size*100)}:{int(test_size*100)}")
    print(f"Epochs run (setosa/versicolor/virginica): {e1}/{e2}/{e3}")
    print("Setosa accuracy:", accuracy(y1_test, y1_pred))
    print("Versicolor accuracy:", accuracy(y2_test, y2_pred))
    print("Virginica accuracy:", accuracy(y3_test, y3_pred))