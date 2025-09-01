import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lab5.lab_utils import get_split_indices, standardize_fit, standardize_apply, random_oversample, binary_metrics, predict_linear_regression, train_linear_regression

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
                w, vloss, *_ = train_linear_regression(X_tr_used, y_tr_used, X_val_s, y_val, lr=0.05, epochs=300, patience=15)
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
