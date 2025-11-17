import pandas as pd
import numpy as np
import os

def _entropy_from_probs(probs):
    """Compute entropy H = -sum p * log_base(p), ignoring p==0."""
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def calculate_entropy_for_column(col, bins=10):
    if col.dtype == 'object' or col.dtype.name == 'category':
        # Categorical variable
        counts = col.dropna().value_counts()
        probs = counts / counts.sum()
        return _entropy_from_probs(probs.values)
    else:
        # Continuous variable: binning
        binned = pd.cut(col.dropna(), bins=bins, duplicates='drop')
        counts = binned.value_counts()
        probs = counts / counts.sum()
        return _entropy_from_probs(probs.values)

def process_dataset(file_path, bins=10):
    df = pd.read_csv(file_path)
    entropies = {}
    for col in df.columns:
        try:
            entropies[col] = calculate_entropy_for_column(df[col], bins)
        except Exception as e:
            entropies[col] = f"Error: {e}"
    return entropies

def main():
    datasets = [
        'lab1/advertising.csv',
        'lab1/Housing.csv',
        'lab2/faa_ai_prelim.csv'
    ]
    results = {}
    for dataset in datasets:
        if os.path.exists(dataset):
            results[dataset] = process_dataset(dataset)
        else:
            results[dataset] = 'File not found.'
    # Save results to file
    with open('lab4/entropy_results.txt', 'w') as f:
        for dataset, entropies in results.items():
            f.write(f"Dataset: {dataset}\n")
            if isinstance(entropies, dict):
                for col, ent in entropies.items():
                    f.write(f"  {col}: {ent}\n")
            else:
                f.write(f"  {entropies}\n")
            f.write("\n")
    print("Entropy results saved to entropy_results.txt")

if __name__ == "__main__":
    main()
