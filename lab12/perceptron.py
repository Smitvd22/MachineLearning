import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def one_hot(y):
    classes, inv = np.unique(y, return_inverse=True)
    K = len(classes)
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), inv] = 1
    return Y, classes


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


class MLP:
    """Very small multilayer perceptron (0/1/2 hidden layers)."""
    def __init__(self, sizes, lr=0.1):
        self.sizes = sizes
        self.lr = lr
        # weights and biases
        self.W = [np.random.randn(a, b) * 0.01 for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]

    def _forward(self, X):
        a = X
        activations = [a]
        zs = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a.dot(W) + b
            zs.append(z)
            if i == len(self.W) - 1:
                a = softmax(z)
            else:
                a = 1 / (1 + np.exp(-z))  # sigmoid
            activations.append(a)
        return activations, zs

    def fit(self, X, Y, epochs=200, batch=16):
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                A, Z = self._forward(X[idx])
                probs = A[-1]
                # backprop
                delta = (probs - Y[idx]) / len(idx)
                dW = []
                db = []
                for l in range(len(self.W) - 1, -1, -1):
                    dW_l = A[l].T.dot(delta)
                    db_l = delta.sum(axis=0, keepdims=True)
                    dW.insert(0, dW_l)
                    db.insert(0, db_l)
                    if l > 0:
                        delta = (delta.dot(self.W[l].T)) * (A[l] * (1 - A[l]))
                # update
                for j in range(len(self.W)):
                    self.W[j] -= self.lr * dW[j]
                    self.b[j] -= self.lr * db[j]

    def predict(self, X):
        A, _ = self._forward(X)
        return np.argmax(A[-1], axis=1)


def load_iris(local=None):
    if local and os.path.exists(local):
        df = pd.read_csv(local)
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
        df = df.rename(columns={
            'SepalLengthCm': 'sepal_length',
            'SepalWidthCm': 'sepal_width',
            'PetalLengthCm': 'petal_length',
            'PetalWidthCm': 'petal_width',
            'Species': 'species'
        })
    else:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        df = pd.read_csv(url, names=['sepal_length','sepal_width','petal_length','petal_width','species'])
    return df


def evaluate_model(sizes, X_train, X_test, y_train, y_test, epochs=300):
    Y_train, classes = one_hot(y_train)
    model = MLP(sizes, lr=0.2)
    model.fit(X_train, Y_train, epochs=epochs)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    acc_train = accuracy(y_train, pred_train)
    acc_test = accuracy(y_test, pred_test)
    return model, acc_train, acc_test


def split(X, y, test_size=0.2, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    t = int(len(X) * test_size)
    test_idx = idx[:t]
    train_idx = idx[t:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    local = r'D:\U23AI118\SEM 5\ML-Lab\lab5\Iris.csv'
    df = load_iris(local)
    X = df.iloc[:, :-1].values.astype(float)
    y = df['species'].values
    # map labels to integers
    classes, inv = np.unique(y, return_inverse=True)
    y_int = inv
    # standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    X_train, X_test, y_train, y_test = split(X, y_int, test_size=0.2, seed=1)

    input_size = X.shape[1]
    n_classes = len(classes)

    configs = {
        '0-hidden': [input_size, n_classes],
        '1-hidden': [input_size, 8, n_classes],
        '2-hidden': [input_size, 12, 6, n_classes]
    }

    results = {}
    for name, sizes in configs.items():
        model, tr, te = evaluate_model(sizes, X_train, X_test, y_train, y_test, epochs=400)
        results[name] = (sizes, tr, te)
        print(f"{name}: train={tr:.4f}, test={te:.4f}")

    # simple bar plot
    names = list(results.keys())
    test_accs = [results[n][2] for n in names]
    plt.bar(names, test_accs)
    plt.ylim(0, 1)
    plt.title('Perceptron variants: test accuracy')
    plt.savefig('perceptron_compact.png', dpi=200)
    print('saved perceptron_compact.png')


if __name__ == '__main__':
    main()
