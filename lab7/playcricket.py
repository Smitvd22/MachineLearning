import csv
import os
import random
from collections import Counter, defaultdict
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), 'playCricket.csv')


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
    def __init__(self, algorithm='ID3', verbose=False):
        assert algorithm in ('ID3', 'C4.5')
        self.algorithm = algorithm
        self.tree = None
        self.verbose = verbose

    def fit(self, rows, attributes, target_attr):
        if self.verbose:
            print(f"Building decision tree using {self.algorithm} algorithm")
            print(f"Available features: {attributes}")
        self.tree = self._build_tree(rows, attributes, target_attr, depth=0, used_features=[])

    def _build_tree(self, rows, attributes, target_attr, depth=0, used_features=None):
        """
        Build decision tree recursively with the constraint that once a feature 
        is used at any level, it cannot be used again at subsequent levels.
        
        Args:
            rows: Training data rows
            attributes: Available attributes for splitting (excludes already used features)
            target_attr: Target attribute name
            depth: Current depth in the tree (for debugging)
            used_features: List of features used from root to current node
        """
        if used_features is None:
            used_features = []
            
        if self.verbose:
            print(f"  {'  ' * depth}Depth {depth}: Available attributes = {attributes}")
            print(f"  {'  ' * depth}Used features so far: {used_features}")
        
        # Check if all instances belong to the same class (pure node)
        classes = [r[target_attr] for r in rows]
        if len(set(classes)) == 1:
            if self.verbose:
                print(f"  {'  ' * depth}Pure node reached: {classes[0]}")
            return {'is_leaf': True, 'label': classes[0], 'used_features': used_features[:]}

        # No more attributes available for splitting
        if not attributes:
            majority = majority_class(rows, target_attr)
            if self.verbose:
                print(f"  {'  ' * depth}No more attributes available, using majority: {majority}")
            return {'is_leaf': True, 'label': majority, 'used_features': used_features[:]}

        # Find the best attribute for splitting
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
            majority = majority_class(rows, target_attr)
            if self.verbose:
                print(f"  {'  ' * depth}No suitable attribute found, using majority: {majority}")
            return {'is_leaf': True, 'label': majority, 'used_features': used_features[:]}

        if self.verbose:
            print(f"  {'  ' * depth}Best attribute: {best_attr} (score: {best_score:.4f})")

        # Create internal node
        tree = {'is_leaf': False, 'attribute': best_attr, 'branches': {}, 'used_features': used_features[:]}
        values = set(r[best_attr] for r in rows)
        
        # Update used features list for child nodes
        new_used_features = used_features + [best_attr]
        
        for v in values:
            subset = [r for r in rows if r[best_attr] == v]
            if not subset:
                majority = majority_class(rows, target_attr)
                tree['branches'][v] = {'is_leaf': True, 'label': majority, 'used_features': new_used_features[:]}
            else:
                # CRITICAL: Remove the used attribute from available attributes
                # This ensures that once a feature is used, it cannot be used again
                # at any subsequent level of the decision tree
                remaining_attrs = [a for a in attributes if a != best_attr]
                
                if self.verbose:
                    print(f"  {'  ' * depth}Splitting on {best_attr}={v}, remaining attributes: {remaining_attrs}")
                
                tree['branches'][v] = self._build_tree(
                    subset, remaining_attrs, target_attr, 
                    depth + 1, new_used_features
                )
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
    
    def print_tree(self, node=None, indent="", branch_value=""):
        """
        Print the decision tree structure showing which features are used at each level.
        This demonstrates that features are not reused once they've been selected.
        """
        if node is None:
            node = self.tree
            print("Decision Tree Structure:")
            print("========================")
        
        if node.get('is_leaf', False):
            used_features = node.get('used_features', [])
            print(f"{indent}└── LEAF: {node['label']} (Features used in path: {used_features})")
        else:
            attr = node['attribute']
            used_features = node.get('used_features', [])
            if branch_value:
                print(f"{indent}└── {branch_value}")
                indent += "    "
            print(f"{indent}Split on: {attr} (Features used so far: {used_features})")
            
            branches = list(node['branches'].items())
            for i, (value, subtree) in enumerate(branches):
                is_last = (i == len(branches) - 1)
                branch_indent = indent + ("    " if is_last else "│   ")
                self.print_tree(subtree, branch_indent, f"{attr} = {value}")
    
    def get_feature_usage_summary(self):
        """
        Get a summary of how features are used in the decision tree.
        Returns information about feature usage at different levels.
        """
        def analyze_node(node, level=0):
            results = {'levels': {}, 'feature_paths': []}
            
            if node.get('is_leaf', False):
                used_features = node.get('used_features', [])
                results['feature_paths'].append(used_features)
            else:
                attr = node['attribute']
                if level not in results['levels']:
                    results['levels'][level] = set()
                results['levels'][level].add(attr)
                
                for subtree in node['branches'].values():
                    sub_results = analyze_node(subtree, level + 1)
                    # Merge results
                    for lvl, features in sub_results['levels'].items():
                        if lvl not in results['levels']:
                            results['levels'][lvl] = set()
                        results['levels'][lvl].update(features)
                    results['feature_paths'].extend(sub_results['feature_paths'])
            
            return results
        
        return analyze_node(self.tree)


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


def run():
    data = load_csv(DATA_PATH)
    attributes = [a for a in data[0].keys() if a not in ('Day', 'PlayCricket')]
    target = 'PlayCricket'

    print("="*60)
    print("DEMONSTRATING FEATURE EXCLUSION IN DECISION TREES")
    print("="*60)
    print(f"Dataset: PlayCricket")
    print(f"Available features: {attributes}")
    print(f"Target: {target}")
    print()

    # Build a verbose decision tree to show feature exclusion
    print("Building Decision Tree with verbose output to show feature exclusion:")
    print("-" * 60)
    
    model = DecisionTree(algorithm='ID3', verbose=True)
    model.fit(data, attributes, target)
    
    print("\n" + "="*60)
    print("DECISION TREE STRUCTURE")
    print("="*60)
    model.print_tree()
    
    print("\n" + "="*60)
    print("FEATURE USAGE ANALYSIS")
    print("="*60)
    usage_summary = model.get_feature_usage_summary()
    
    print("Features used at each level:")
    for level, features in sorted(usage_summary['levels'].items()):
        print(f"  Level {level}: {list(features)}")
    
    print(f"\nFeature usage paths (root to leaf):")
    for i, path in enumerate(usage_summary['feature_paths'], 1):
        print(f"  Path {i}: {path}")
    
    # Verify no feature is reused in any path
    print(f"\nVerification - No feature reused in any path:")
    all_paths_unique = True
    for i, path in enumerate(usage_summary['feature_paths'], 1):
        if len(path) != len(set(path)):
            print(f"  ❌ Path {i} has repeated features: {path}")
            all_paths_unique = False
        else:
            print(f"  ✅ Path {i} has unique features: {path}")
    
    if all_paths_unique:
        print(f"\n🎉 SUCCESS: All paths have unique features - constraint satisfied!")
    else:
        print(f"\n❌ ERROR: Some paths have repeated features - constraint violated!")

    print("\n" + "="*60)
    print("CROSS-VALIDATION PERFORMANCE")
    print("="*60)

    folds = k_fold_split(data, 5)

    for alg in ('ID3', 'C4.5'):
        scores = []
        for i in range(5):
            test = folds[i]
            train = [r for j, f in enumerate(folds) if j != i for r in f]
            model = DecisionTree(algorithm=alg, verbose=False)  # Turn off verbose for CV
            model.fit(train, attributes, target)
            default = majority_class(train, target)
            y_true = [r[target] for r in test]
            y_pred = [model.predict_row(r, default=default) for r in test]
            m = metrics_classification(y_true, y_pred)
            scores.append(m)
        avg = {k: sum(d[k] for d in scores) / len(scores) for k in scores[0]}
        print(f"Algorithm={alg} -> accuracy={avg['accuracy']:.4f}, precision_macro={avg['precision_macro']:.4f}, recall_macro={avg['recall_macro']:.4f}, f1_macro={avg['f1_macro']:.4f}")

    # Train sklearn DecisionTreeClassifier on full dataset and plot
    print(f"\nGenerating sklearn comparison plot...")
    X = [[r[a] for a in attributes] for r in data]
    enc = OrdinalEncoder()
    X_enc = enc.fit_transform(X)
    y = [r[target] for r in data]

    sk_clf = DecisionTreeClassifier(max_depth=6, random_state=1)
    sk_clf.fit(X_enc, y)

    plt.figure(figsize=(12, 8))
    plot_tree(sk_clf, feature_names=attributes, class_names=sk_clf.classes_, filled=True, rounded=True)
    plt.title("Decision Tree (sklearn) trained on playCricket.csv\n(Note: sklearn may reuse features, unlike our implementation)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
