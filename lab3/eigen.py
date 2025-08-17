import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def reconstruction_error(X, n_components_list):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    mu = np.mean(X_std, axis=0)
    A = X_std - mu
    ATA = np.dot(A.T, A)
    eigvals, eigvecs = np.linalg.eigh(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    errors = []
    for n in n_components_list:
        top_eigvecs = eigvecs[:, :n]
        W = np.dot(A, top_eigvecs)
        L = np.dot(W, top_eigvecs.T)
        R = L + mu
        error = np.sum((X_std - R) ** 2)
        errors.append(error)
    return errors

def plot_errors(errors_list, n_components_list, dataset_names):
    plt.figure(figsize=(10,7))
    for errors, name, comps in zip(errors_list, dataset_names, n_components_list):
        plt.plot(comps, errors, marker='o', label=f"{name} (max {max(comps)})")
    plt.xlabel('Number of Eigenvectors')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Number of Eigenvectors')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

csv_paths = [
    r'lab1/advertising.csv',
    r'lab1/Housing.csv',
    r'lab2/faa_ai_prelim.csv'
]
dataset_names = ['Advertising', 'Housing', 'FAA']
errors_list = []
dataframes = []
n_components_lists = []
for path in csv_paths:
    df = pd.read_csv(path)
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.dropna(axis=0)
    if df_numeric.shape[0] == 0 or df_numeric.shape[1] == 0:
        print(f"Warning: No valid numeric data in {path} after cleaning. Skipping.")
        continue
    if df_numeric.shape[1] > 1:
        X = df_numeric.iloc[:, :-1].values
    else:
        X = df_numeric.values
    # Dynamically set n_components based on feature count
    max_comp = min(15, X.shape[1])
    n_components_list = list(range(1, max_comp+1))
    n_components_lists.append(n_components_list)
    dataframes.append((X, df_numeric))
    errors = reconstruction_error(X, n_components_list)
    errors_list.append(errors)

plot_errors(errors_list, n_components_lists, dataset_names[:len(errors_list)])

# --- Gradient Descent Algorithms ---
def simple_gradient_descent(X, y, lr=0.001, epochs=100):
    # Normalize X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y.reshape(-1,1)).flatten()
    w = np.zeros(X_norm.shape[1])
    losses = []
    for _ in range(epochs):
        grad = 2 * np.dot(X_norm.T, np.dot(X_norm, w) - y_norm) / len(y_norm)
        w -= lr * grad
        loss = np.mean((np.dot(X_norm, w) - y_norm) ** 2)
        losses.append(loss)
    return w, losses

def momentum_gradient_descent(X, y, lr=0.001, gamma=0.9, epochs=100):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y.reshape(-1,1)).flatten()
    w = np.zeros(X_norm.shape[1])
    v = np.zeros_like(w)
    losses = []
    for _ in range(epochs):
        grad = 2 * np.dot(X_norm.T, np.dot(X_norm, w) - y_norm) / len(y_norm)
        v = gamma * v + lr * grad
        w -= v
        loss = np.mean((np.dot(X_norm, w) - y_norm) ** 2)
        losses.append(loss)
    return w, losses

def nesterov_gradient_descent(X, y, lr=0.001, gamma=0.9, epochs=100):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y.reshape(-1,1)).flatten()
    w = np.zeros(X_norm.shape[1])
    v = np.zeros_like(w)
    losses = []
    for _ in range(epochs):
        lookahead_w = w - gamma * v
        grad = 2 * np.dot(X_norm.T, np.dot(X_norm, lookahead_w) - y_norm) / len(y_norm)
        v = gamma * v + lr * grad
        w -= v
        loss = np.mean((np.dot(X_norm, w) - y_norm) ** 2)
        losses.append(loss)
    return w, losses

def compare_gd_algorithms(X, y, dataset_name):
    w1, losses1 = simple_gradient_descent(X, y)
    w2, losses2 = momentum_gradient_descent(X, y)
    w3, losses3 = nesterov_gradient_descent(X, y)
    plt.figure(figsize=(8,6))
    plt.plot(losses1, label='Simple GD')
    plt.plot(losses2, label='Momentum GD')
    plt.plot(losses3, label='Nesterov GD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'GD Algorithms Comparison on {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

for (X, df_numeric), name in zip(dataframes, dataset_names[:len(dataframes)]):
    # Use last column as target if >1 columns, else mean of X
    if df_numeric.shape[1] > 1:
        y = df_numeric.iloc[:, -1].values
    else:
        y = np.mean(X, axis=1)
    compare_gd_algorithms(X, y, name)