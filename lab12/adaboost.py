"""
Multiclass AdaBoost pipeline for lab12/Score.csv

Steps:
- Load dataset `Score.csv` (expects it in the same folder)
- Create train/validation/test splits (80/10/10)
- Preprocess numeric and categorical features
- Train AdaBoost (SAMME) with decision stumps and select n_estimators by validation
- Evaluate on test set and print classification report + confusion matrix

Run: python adaboost.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import local implementation
from SAMME import SAMMEClassifier


DATA_PATH = os.path.join(os.path.dirname(__file__), "Score.csv")


def load_data(path=DATA_PATH):
	df = pd.read_csv(path)
	return df


def prepare_splits(df, label_col="Credit_Score", random_state=42):
	X = df.drop(columns=[label_col])
	y = df[label_col]

	# First split: train (80%) and temp (20%)
	X_train, X_temp, y_train, y_temp = train_test_split(
		X, y, test_size=0.2, random_state=random_state, stratify=y
	)

	# Split temp into val and test equally -> each 10% of original
	X_val, X_test, y_val, y_test = train_test_split(
		X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
	)

	return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X):
	# detect categorical and numeric columns
	cat_cols = X.select_dtypes(include=[object]).columns.tolist()
	num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

	num_transformer = StandardScaler()
	# use sparse_output for compatibility with different sklearn versions
	try:
		cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		# fallback for older sklearn versions
		cat_transformer = OneHotEncoder(handle_unknown="ignore")

	preprocessor = ColumnTransformer(
		transformers=[
			("num", num_transformer, num_cols),
			("cat", cat_transformer, cat_cols),
		]
	)
	return preprocessor


def train_and_select(X_train, y_train, X_val, y_val, preprocessor, model_name="sklearn-SAMME",
					 algorithm=None, random_state=42):
	"""
	Generalized training + validation selection for n_estimators.
	model_name: one of 'sklearn-SAMME', 'sklearn-SAMME.R', 'custom-SAMME'
	algorithm: for sklearn variants, the algorithm string to pass (e.g. 'SAMME' or 'SAMME.R')
	"""
	candidates = [10, 50, 100, 200]
	best = None
	best_acc = -1
	best_model = None

	for n in candidates:
		if model_name.startswith("sklearn"):
			stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
			# construct AdaBoost in a way compatible with several sklearn versions
			try:
				clf = AdaBoostClassifier(
					base_estimator=stump,
					n_estimators=n,
					algorithm=algorithm,
					random_state=random_state,
				)
			except Exception:
				try:
					clf = AdaBoostClassifier(
						estimator=stump,
						n_estimators=n,
						algorithm=algorithm,
						random_state=random_state,
					)
				except Exception:
					# fallback: construct without 'algorithm' (let sklearn default)
					try:
						clf = AdaBoostClassifier(
							base_estimator=stump,
							n_estimators=n,
							random_state=random_state,
						)
					except Exception:
						clf = AdaBoostClassifier(
							estimator=stump,
							n_estimators=n,
							random_state=random_state,
						)

			pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
			try:
				pipe.fit(X_train, y_train)
			except Exception:
				# If sklearn in this environment rejects the 'algorithm' value at fit time,
				# fall back to constructing without algorithm param and retry.
				try:
					clf = AdaBoostClassifier(
						base_estimator=stump,
						n_estimators=n,
						random_state=random_state,
					)
				except Exception:
					clf = AdaBoostClassifier(
						estimator=stump,
						n_estimators=n,
						random_state=random_state,
					)
				pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
				pipe.fit(X_train, y_train)
		else:
			# custom SAMME implementation expects raw X and y (we'll apply preprocessor separately)
			clf = SAMMEClassifier(n_estimators=n, random_state=random_state,
								  base_estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state))
			# fit on preprocessed arrays
			Xtr = preprocessor.fit_transform(X_train)
			clf.fit(Xtr, y_train.values)
			# wrap to mimic sklearn pipeline predict
			class Wrapped:
				def __init__(self, pre, model):
					self.pre = pre
					self.model = model

				def predict(self, X):
					Xt = self.pre.transform(X)
					return self.model.predict(Xt)

			pipe = Wrapped(preprocessor, clf)

		preds = pipe.predict(X_val)
		acc = accuracy_score(y_val, preds)
		print(f"{model_name} n_estimators={n} validation accuracy: {acc:.4f}")

		if acc > best_acc:
			best_acc = acc
			best = n
			best_model = pipe

	print(f"Selected for {model_name} n_estimators={best} with val acc={best_acc:.4f}")
	return best_model, best


def evaluate(model, X_test, y_test):
	preds = model.predict(X_test)
	acc = accuracy_score(y_test, preds)
	print("\nTest Accuracy: ", acc)
	print("\nClassification Report:\n")
	print(classification_report(y_test, preds))
	print("\nConfusion Matrix:\n")
	print(confusion_matrix(y_test, preds))


def main():
	print("Loading data from:", DATA_PATH)
	df = load_data()

	# Basic cleanup: drop rows with NA in label
	df = df.dropna(subset=["Credit_Score"]).reset_index(drop=True)

	X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df)

	# build preprocessor once
	preprocessor = build_preprocessor(X_train)

	# 1) sklearn AdaBoost (SAMME.R - real boosting)
	model1, n1 = train_and_select(X_train, y_train, X_val, y_val, preprocessor,
								  model_name="sklearn-SAMME.R", algorithm="SAMME.R")

	# 2) sklearn AdaBoost (SAMME - discrete)
	model2, n2 = train_and_select(X_train, y_train, X_val, y_val, preprocessor,
								  model_name="sklearn-SAMME", algorithm="SAMME")

	# 3) custom SAMME implementation
	model3, n3 = train_and_select(X_train, y_train, X_val, y_val, preprocessor,
								  model_name="custom-SAMME", algorithm=None)

	print("\nEvaluating on test set: sklearn SAMME.R")
	evaluate(model1, X_test, y_test)

	print("\nEvaluating on test set: sklearn SAMME (discrete)")
	evaluate(model2, X_test, y_test)

	print("\nEvaluating on test set: custom SAMME")
	evaluate(model3, X_test, y_test)


if __name__ == "__main__":
	main()

