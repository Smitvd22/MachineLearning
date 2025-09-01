import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lab5.lab_utils import get_split_indices, standardize_fit, standardize_apply, predict_linear_regression, train_linear_regression

def train_linear_regression(X_train, y_train, X_val, y_val, lr=0.01, epochs=500, patience=20):
	# simple linear regression with early stopping on validation MSE
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

	return best_w

def multiclass_metrics(y_true, y_pred, labels):
	# labels: list of label values
	# compute per-class TP, FP, FN
	stats = {}
	total_tp = total_fp = total_fn = 0
	supports = {}
	for cls in labels:
		tp = int(((y_true == cls) & (y_pred == cls)).sum())
		fp = int(((y_true != cls) & (y_pred == cls)).sum())
		fn = int(((y_true == cls) & (y_pred != cls)).sum())
		supports[cls] = int((y_true == cls).sum())
		total_tp += tp
		total_fp += fp
		total_fn += fn
		prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
		stats[cls] = {'tp': tp, 'fp': fp, 'fn': fn, 'prec': prec, 'rec': rec, 'f1': f1}

	# macro: average of per-class f1
	macro_f1 = np.mean([stats[cls]['f1'] for cls in labels])

	# micro: compute from aggregated counts
	micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
	micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
	micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

	# weighted: weighted average of per-class f1 by support
	total_support = sum(supports.values())
	weighted_f1 = 0.0
	if total_support > 0:
		weighted_f1 = sum(stats[cls]['f1'] * supports[cls] for cls in labels) / total_support

	return stats, {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1}

def run_experiment():
	# load iris from lab5 folder
	df = pd.read_csv('lab5/Iris.csv')
	X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy(dtype=float)
	y_str = df['Species'].to_numpy()
	# map species to integers
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

	# train one-vs-rest linear regression for each class
	n_classes = len(classes)
	Ws = []
	for cls_idx in range(n_classes):
		ytr_bin = (y_train == cls_idx).astype(float)
		yval_bin = (y_val == cls_idx).astype(float)
		w = train_linear_regression(X_train_s, ytr_bin, X_val_s, yval_bin, lr=0.05, epochs=1000, patience=30)
		Ws.append(w)

	# predict: compute scores for each class and pick argmax
	scores = np.vstack([predict_linear_regression(w, X_test_s) for w in Ws]).T  # shape (n_samples, n_classes)
	y_pred = np.argmax(scores, axis=1)

	# compute metrics
	stats, agg = multiclass_metrics(y_test, y_pred, list(range(n_classes)))

	# print results
	print('Per-class metrics (class index -> original label):')
	for i, label in enumerate(classes):
		s = stats[i]
		print(f" Class {i} ({label}): support={int((y_test==i).sum())}  prec={s['prec']:.3f}  rec={s['rec']:.3f}  f1={s['f1']:.3f}")

	print('\nAggregated F1 scores:')
	print(f" Macro F1:     {agg['macro_f1']:.4f}")
	print(f" Micro F1:     {agg['micro_f1']:.4f}")
	print(f" Weighted F1:  {agg['weighted_f1']:.4f}")

if __name__ == '__main__':
	run_experiment()

