import csv
import os
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

DATA_PATH = os.path.join(os.path.dirname(__file__), 'petrol_consumption.csv')


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def mse(y_true, y_pred):
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, rows, features, target, depth=0):
        self.tree = self._build_tree(rows, features, target, depth)

    def _variance(self, rows, target):
        vals = [float(r[target]) for r in rows]
        mean = sum(vals) / len(vals)
        return sum((v - mean) ** 2 for v in vals) / len(vals)

    def _best_split(self, rows, features, target):
        best_feat = None
        best_thresh = None
        best_var_red = 0
        base_var = self._variance(rows, target)
        n = len(rows)
        for feat in features:
            vals = sorted(set(float(r[feat]) for r in rows))
            if len(vals) <= 1:
                continue
            thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals)-1)]
            for t in thresholds:
                left = [r for r in rows if float(r[feat]) <= t]
                right = [r for r in rows if float(r[feat]) > t]
                if not left or not right:
                    continue
                var_left = self._variance(left, target)
                var_right = self._variance(right, target)
                remainder = (len(left)/n)*var_left + (len(right)/n)*var_right
                var_red = base_var - remainder
                if var_red > best_var_red:
                    best_var_red = var_red
                    best_feat = feat
                    best_thresh = t
        return best_feat, best_thresh, best_var_red

    def _build_tree(self, rows, features, target, depth):
        if len(rows) < self.min_samples_split or depth >= self.max_depth:
            vals = [float(r[target]) for r in rows]
            return {'is_leaf': True, 'value': sum(vals) / len(vals)}

        feat, thresh, var_red = self._best_split(rows, features, target)
        if feat is None or var_red <= 0:
            vals = [float(r[target]) for r in rows]
            return {'is_leaf': True, 'value': sum(vals) / len(vals)}

        left = [r for r in rows if float(r[feat]) <= thresh]
        right = [r for r in rows if float(r[feat]) > thresh]
        node = {'is_leaf': False, 'feature': feat, 'threshold': thresh,
                'left': self._build_tree(left, features, target, depth+1),
                'right': self._build_tree(right, features, target, depth+1)}
        return node

    def predict_row(self, row, node=None):
        if node is None:
            node = self.tree
        if node.get('is_leaf'):
            return node['value']
        feat = node['feature']
        thresh = node['threshold']
        if float(row[feat]) <= thresh:
            return self.predict_row(row, node['left'])
        else:
            return self.predict_row(row, node['right'])

    def predict(self, rows):
        return [self.predict_row(r) for r in rows]


def k_fold_split(rows, k=5, seed=1):
    rows_copy = list(rows)
    random.Random(seed).shuffle(rows_copy)
    n = len(rows_copy)
    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else (i + 1) * fold_size
        folds.append(rows_copy[start:end])
    return folds


def run():
    data = load_csv(DATA_PATH)
    features = [f for f in data[0].keys() if f != 'Petrol_Consumption']
    target = 'Petrol_Consumption'
    folds = k_fold_split(data, 5)

    scores = []
    for i in range(5):
        test = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        model = RegressionTree(min_samples_split=5, max_depth=6)
        model.fit(train, features, target)
        y_true = [float(r[target]) for r in test]
        y_pred = model.predict(test)
        scores.append(mse(y_true, y_pred))
    avg_mse = sum(scores) / len(scores)
    print(f"Decision tree regression on petrol_consumption.csv -> MSE={avg_mse:.4f}")

    # Train an sklearn DecisionTreeRegressor on the full dataset and plot it
    sk_model = DecisionTreeRegressor(min_samples_split=5, max_depth=6, random_state=1)
    X = [[float(r[f]) for f in features] for r in data]
    y = [float(r[target]) for r in data]
    sk_model.fit(X, y)

    plt.figure(figsize=(18, 8))
    plot_tree(sk_model, feature_names=features, filled=True, rounded=True)
    plt.title("Decision Tree (sklearn) trained on petrol_consumption.csv")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
