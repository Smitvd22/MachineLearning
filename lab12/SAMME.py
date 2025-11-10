"""
Simple from-scratch SAMME (discrete multiclass AdaBoost) implementation.

This implementation uses scikit-learn compatible DecisionTreeClassifier stumps
as weak learners and supports fit/predict interface.

Note: This is an educational implementation and not optimized for performance.
"""
from typing import List
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SAMMEClassifier:
    def __init__(self, n_estimators=50, base_estimator=None, random_state=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.classes_ = None
        self.estimators_: List[DecisionTreeClassifier] = []
        self.alphas_: List[float] = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        K = len(self.classes_)

        # initialize weights
        sample_weights = np.full(n_samples, 1.0 / n_samples)

        # default base estimator: decision stump
        for m in range(self.n_estimators):
            stump = (
                DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
                if self.base_estimator is None
                else self.base_estimator
            )

            # fit with sample weights
            stump.fit(X, y, sample_weight=sample_weights)
            pred = stump.predict(X)
            incorrect = (pred != y).astype(float)

            # weighted error
            err_m = np.dot(sample_weights, incorrect) / sample_weights.sum()

            # avoid divide by zero
            if err_m <= 0:
                alpha_m = 1.0
                self.estimators_.append(stump)
                self.alphas_.append(alpha_m)
                break

            # If error is too large, break (can't improve)
            if err_m >= 1.0 - (1.0 / K):
                # skip this estimator
                break

            # SAMME alpha
            alpha_m = np.log((1 - err_m) / err_m) + np.log(K - 1)

            # update weights
            sample_weights = sample_weights * np.exp(alpha_m * incorrect)
            sample_weights = sample_weights / sample_weights.sum()

            self.estimators_.append(stump)
            self.alphas_.append(alpha_m)

        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self.estimators_:
            raise ValueError("No estimators fitted. Call fit first.")

        # accumulate votes
        classes = self.classes_
        K = len(classes)
        n_samples = X.shape[0]
        agg = np.zeros((n_samples, K))

        for alpha, est in zip(self.alphas_, self.estimators_):
            preds = est.predict(X)
            # map preds to class indices
            idxs = np.searchsorted(classes, preds)
            # add alpha to corresponding class column
            for i in range(n_samples):
                agg[i, idxs[i]] += alpha

        # choose class with max aggregated score
        chosen = agg.argmax(axis=1)
        return classes[chosen]
