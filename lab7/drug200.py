import csv
import os
import random
from collections import Counter, defaultdict
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), 'drug_200.csv')


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def entropy(rows, target_attr):
    counts = Counter(r[target_attr] for r in rows)
    total = len(rows)
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def information_gain(rows, attr, target_attr):
    base = entropy(rows, target_attr)
    total = len(rows)
    subsets = defaultdict(list)
    for r in rows:
        subsets[r[attr]].append(r)
    remainder = 0.0
    for sv in subsets.values():
        remainder += (len(sv) / total) * entropy(sv, target_attr)
    return base - remainder


def split_info(rows, attr):
    total = len(rows)
    counts = Counter(r[attr] for r in rows)
    si = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            si -= p * math.log2(p)
    return si


def gain_ratio(rows, attr, target_attr):
    ig = information_gain(rows, attr, target_attr)
    si = split_info(rows, attr)
    if si == 0:
        return 0
    return ig / si


def majority_class(rows, target_attr):
    return Counter(r[target_attr] for r in rows).most_common(1)[0][0]


class DecisionTree:
    def __init__(self, algorithm='ID3'):
        assert algorithm in ('ID3', 'C4.5')
        self.algorithm = algorithm
        self.tree = None

    def fit(self, rows, attributes, target_attr):
        self.tree = self._build_tree(rows, attributes, target_attr)

    def _build_tree(self, rows, attributes, target_attr):
        classes = [r[target_attr] for r in rows]
        if len(set(classes)) == 1:
            return {'is_leaf': True, 'label': classes[0]}

        if not attributes:
            return {'is_leaf': True, 'label': majority_class(rows, target_attr)}

        best_attr = None
        best_score = -1
        for attr in attributes:
            if self.algorithm == 'ID3':
                score = information_gain(rows, attr, target_attr)
            else:
                score = gain_ratio(rows, attr, target_attr)
            if score > best_score:
                best_score = score
                best_attr = attr

        if best_attr is None:
            return {'is_leaf': True, 'label': majority_class(rows, target_attr)}

        tree = {'is_leaf': False, 'attribute': best_attr, 'branches': {}}
        values = set(r[best_attr] for r in rows)
        for v in values:
            subset = [r for r in rows if r[best_attr] == v]
            if not subset:
                tree['branches'][v] = {'is_leaf': True, 'label': majority_class(rows, target_attr)}
            else:
                new_attrs = [a for a in attributes if a != best_attr]
                tree['branches'][v] = self._build_tree(subset, new_attrs, target_attr)
        return tree

    def predict_row(self, row, default=None):
        node = self.tree
        while not node.get('is_leaf', False):
            attr = node['attribute']
            val = row.get(attr)
            node = node['branches'].get(val)
            if node is None:
                return default
        return node['label']


def metrics_classification(y_true, y_pred):
    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / total
    labels = set(y_true) | set(y_pred)
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {'accuracy': accuracy, 'precision_macro': sum(precisions) / len(precisions),
            'recall_macro': sum(recalls) / len(recalls), 'f1_macro': sum(f1s) / len(f1s)}


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


def convert_continuous_to_bool(rows, column, threshold=None):
    # If threshold not provided, use median
    vals = [float(r[column]) for r in rows]
    if threshold is None:
        sorted_vals = sorted(vals)
        m = len(sorted_vals)
        threshold = (sorted_vals[m//2] if m % 2 == 1 else (sorted_vals[m//2 - 1] + sorted_vals[m//2]) / 2)
    for r in rows:
        r[column + '_high'] = 'True' if float(r[column]) > threshold else 'False'
    return column + '_high'


def run():
    data = load_csv(DATA_PATH)
    # convert Na_to_K continuous into boolean
    bool_col = convert_continuous_to_bool(data, 'Na_to_K')

    attributes = [a for a in data[0].keys() if a not in ('Age', 'Na_to_K', 'Drug')]
    attributes.append(bool_col)
    target = 'Drug'

    folds = k_fold_split(data, 5)

    for alg in ('ID3', 'C4.5'):
        scores = []
        for i in range(5):
            test = folds[i]
            train = [r for j, f in enumerate(folds) if j != i for r in f]
            model = DecisionTree(algorithm=alg)
            model.fit(train, attributes, target)
            default = majority_class(train, target)
            y_true = [r[target] for r in test]
            y_pred = [model.predict_row(r, default=default) for r in test]
            m = metrics_classification(y_true, y_pred)
            scores.append(m)
        avg = {k: sum(d[k] for d in scores) / len(scores) for k in scores[0]}
        print(f"Algorithm={alg} on drug_200.csv -> accuracy={avg['accuracy']:.4f}, precision_macro={avg['precision_macro']:.4f}, recall_macro={avg['recall_macro']:.4f}, f1_macro={avg['f1_macro']:.4f}")

    # Train sklearn DecisionTreeClassifier on full dataset and plot
    X = [[r[a] for a in attributes] for r in data]
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    y = [r[target] for r in data]

    sk_clf = DecisionTreeClassifier(max_depth=6, random_state=1)
    sk_clf.fit(X_enc, y)

    plt.figure(figsize=(18, 8))
    plot_tree(sk_clf, feature_names=attributes, class_names=sk_clf.classes_, filled=True, rounded=True)
    plt.title("Decision Tree (sklearn) trained on drug_200.csv")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
