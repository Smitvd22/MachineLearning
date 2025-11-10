"""
Simple AdaBoost implementation (from-scratch) for the sklearn breast cancer dataset.

This script:
- loads the sklearn breast cancer dataset
- splits it into train/validation/test in 80/10/10 ratio (stratified)
- trains AdaBoost with decision stumps as weak learners
- evaluates and prints accuracy on train/val/test
- writes test predictions to `test_predictions_adaboost.csv` in the same folder

Usage:
	python "d:\\U23AI118\\SEM 5\\ML-Lab\\lab11\\adaboost.py"

"""

from __future__ import annotations
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class DecisionStump:
	feature_index: int = 0
	threshold: float = 0.0
	polarity: int = 1  # 1 or -1

	def predict(self, X: np.ndarray) -> np.ndarray:
		# X: (n_samples, n_features)
		n = X.shape[0]
		preds = np.ones(n, dtype=int)
		# apply polarity: if polarity==1 then predict -1 when x < thresh else +1
		if self.polarity == 1:
			preds[X[:, self.feature_index] < self.threshold] = -1
		else:
			preds[X[:, self.feature_index] >= self.threshold] = -1
		return preds


class AdaBoost:
	def __init__(self, n_estimators: int = 50):
		self.n_estimators = n_estimators
		self.learners: List[DecisionStump] = []
		self.alphas: List[float] = []

	def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> None:
		n_samples, n_features = X.shape
		# convert y to {-1, +1}
		y_adj = y.copy().astype(int)
		unique = np.unique(y_adj)
		if set(unique) <= {0, 1}:
			y_adj = np.where(y_adj == 0, -1, 1)

		# initialize weights
		if sample_weights is None:
			w = np.full(n_samples, 1 / n_samples)
		else:
			w = sample_weights.astype(float)
			w = w / np.sum(w)

		for m in range(self.n_estimators):
			stump = self._fit_stump(X, y_adj, w)
			preds = stump.predict(X)
			# weighted error
			incorrect = preds != y_adj
			err = np.dot(w, incorrect)  # sum(w_i * I(h(x_i) != y_i))

			# prevent division by zero or perfect fit producing infinite alpha
			err = max(1e-10, min(err, 1 - 1e-10))

			alpha = 0.5 * math.log((1 - err) / err)

			# update weights
			w = w * np.exp(-alpha * y_adj * preds)
			w = w / np.sum(w)

			self.learners.append(stump)
			self.alphas.append(alpha)

			# early stopping if error is 0 (perfect)
			if err == 0:
				break

	def _fit_stump(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> DecisionStump:
		n_samples, n_features = X.shape
		best_stump = DecisionStump()
		best_error = float('inf')

		for feature_i in range(n_features):
			feature_values = X[:, feature_i]
			# candidate thresholds: midpoints between sorted unique values
			uniq_vals = np.unique(feature_values)
			if len(uniq_vals) == 1:
				thresholds = uniq_vals
			else:
				thresholds = (uniq_vals[:-1] + uniq_vals[1:]) / 2.0

			for thresh in thresholds:
				for polarity in (1, -1):
					stump = DecisionStump(feature_index=feature_i, threshold=thresh, polarity=polarity)
					preds = stump.predict(X)
					error = np.dot(w, preds != y)
					if error < best_error:
						best_error = error
						best_stump = stump

		return best_stump

	def predict(self, X: np.ndarray) -> np.ndarray:
		# aggregate predictions
		if not self.learners:
			raise ValueError("Model not fitted")
		agg = np.zeros(X.shape[0], dtype=float)
		for alpha, learner in zip(self.alphas, self.learners):
			agg += alpha * learner.predict(X)
		return np.sign(agg).astype(int)


def load_and_split(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X, y = load_breast_cancer(return_X_y=True)
	# first split: train (80%) and temp (20%)
	X_train, X_temp, y_train, y_temp = train_test_split(
		X, y, test_size=0.2, stratify=y, random_state=random_state
	)
	# split temp into validation and test (each 10% of total => half of temp)
	X_val, X_test, y_val, y_test = train_test_split(
		X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
	)
	return X_train, y_train, X_val, y_val, X_test, y_test


def save_test_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
	# writes CSV with header: true,pred
	with open(path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['true', 'pred'])
		for t, p in zip(y_true.tolist(), y_pred.tolist()):
			writer.writerow([int(t), int(p)])


def main():
	X_train, y_train, X_val, y_val, X_test, y_test = load_and_split(random_state=1)

	model = AdaBoost(n_estimators=50)
	model.fit(X_train, y_train)

	# predictions: our predict returns {-1, +1}, but original labels are {0,1}
	def convert_back(arr: np.ndarray) -> np.ndarray:
		# arr is in {-1, +1}
		return np.where(arr == -1, 0, 1)

	# evaluate on train, val, test
	y_train_pred = convert_back(model.predict(X_train))
	y_val_pred = convert_back(model.predict(X_val))
	y_test_pred = convert_back(model.predict(X_test))

	train_acc = accuracy_score(y_train, y_train_pred)
	val_acc = accuracy_score(y_val, y_val_pred)
	test_acc = accuracy_score(y_test, y_test_pred)

	print(f"Train accuracy: {train_acc:.4f}")
	print(f"Validation accuracy: {val_acc:.4f}")
	print(f"Test accuracy: {test_acc:.4f}")

	out_path = 'd:\\U23AI118\\SEM 5\\ML-Lab\\lab11\\test_predictions_adaboost.csv'
	save_test_predictions(y_test, y_test_pred, out_path)
	print(f"Wrote test predictions to: {out_path}")


if __name__ == '__main__':
	main()

