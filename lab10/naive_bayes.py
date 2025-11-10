import os
from pathlib import Path
import numpy as np
import pandas as pd


def find_bank_csv():
    """Try a list of candidate paths to locate bank-full.csv."""
    # Known absolute path in this workspace (use this directly)
    known_path = Path(r"D:\U23AI118\SEM 5\ML-Lab\lab9\Assignment-9\bank-full.csv")
    if known_path.exists():
        return str(known_path)

    # Fallback: try the local project locations (rarely used)
    candidates = [
        Path(__file__).parent / "bank-full.csv",
        Path(__file__).parent.parent / "lab9" / "Assignment-9" / "bank-full.csv",
        Path("lab9") / "Assignment-9" / "bank-full.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"Could not find 'bank-full.csv'. Tried: {known_path} and local candidates.\n"
        "Please place the file at the known path or update the script."
    )


def encode_categorical_columns(df):
    """
    Encode object-type columns into integer codes using a reproducible mapping.
    Returns the encoded dataframe and a dict of mappings for each encoded column.
    """
    encoders = {}
    for col in df.columns:
        # Use isinstance check to avoid deprecated is_categorical_dtype
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            uniques = list(df[col].unique())
            mapping = {val: idx for idx, val in enumerate(uniques)}
            df[col] = df[col].map(mapping)
            encoders[col] = mapping
    return df, encoders


class NaiveBayesScratch:
    """Gaussian Naive Bayes implemented from scratch."""

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            # add small epsilon to variance to avoid division by zero
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.priors[c] = len(X_c) / n

    def _gaussian_logpdf(self, x, mean, var):
        """Return log of Gaussian PDF for numeric stability."""
        # log(1/sqrt(2*pi*var)) - ((x-mean)^2)/(2*var)
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict_single(self, x):
        posteriors = {}
        for c in self.classes:
            prior_log = np.log(self.priors[c])
            # sum log pdfs across features (assume independence)
            likelihood_log = np.sum(self._gaussian_logpdf(x, self.mean[c], self.var[c]))
            posteriors[c] = prior_log + likelihood_log
        # return class with highest posterior log-probability
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


def compute_confusion_binary(y_true, y_pred, positive_label=1):
    TP = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    FP = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    TN = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    FN = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    return TP, FP, TN, FN


def compute_metrics_from_confusion(TP, FP, TN, FN):
    eps = 1e-9
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return accuracy, precision, recall, f1


def main():
    # Locate CSV
    csv_path = find_bank_csv()
    print(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path, sep=';', engine='python', on_bad_lines='warn')

    # Encode categorical columns without sklearn
    df_encoded, encoders = encode_categorical_columns(df.copy())

    print('\nColumns encoded and their mapping samples:')
    for col, mapping in encoders.items():
        sample_items = list(mapping.items())[:5]
        print(f"  {col}: {sample_items} {'...' if len(mapping)>5 else ''}")

    # Shuffle using fixed seed to get same split as in Assignment 9
    df_shuffled = df_encoded.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract features and target
    X = df_shuffled.drop("y", axis=1).values
    y = df_shuffled["y"].values

    split = int(0.8 * len(df_shuffled))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset size: {len(df_shuffled)} samples")
    print(f"Train / Test split: {len(X_train)} / {len(X_test)}")

    # Train model
    model = NaiveBayesScratch()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Compute confusion and metrics (binary positive assumed to be encoded as 1)
    # Find which label corresponds to original 'yes' if encoders contains 'y'
    positive_label = 1
    if 'y' in encoders:
        # we don't know mapping order; try to find which original value was 'yes' or 'y'
        inv_map = {v: k for k, v in encoders['y'].items()}
        # If original labels were 'yes'/'no', attempt to find which code is 'yes'
        for code, orig in inv_map.items():
            if str(orig).lower().startswith('y') or str(orig).lower() == 'yes':
                positive_label = code
                break

    TP, FP, TN, FN = compute_confusion_binary(y_test, y_pred, positive_label=positive_label)
    accuracy, precision, recall, f1 = compute_metrics_from_confusion(TP, FP, TN, FN)

    print("\nConfusion matrix (positive label =", positive_label, "):")
    print(f"  TP: {TP}\n  FP: {FP}\n  TN: {TN}\n  FN: {FN}")
    print("\nEvaluation metrics:")
    print(f"  Accuracy : {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall   : {recall:.6f}")
    print(f"  F1-score : {f1:.6f}")


if __name__ == "__main__":
    main()