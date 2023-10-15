# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

LOKY_MAX_CPU_COUNT = 4

# ---DATASET--------------------------------------------------------------------
df = pd.read_csv("Regression_datasets/advertising.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1:].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# ---MODEL----------------------------------------------------------------------
class LinearRegression:
    def __init__(self, name, scaler=None):
        self.name = name
        self.scaler = scaler
        self.weights = None

    def _build(self, X):
        self.weights = np.zeros((X.shape[1] + 1))
        if self.scaler is not None:
            self.scaler.fit(X)

    def _loss(self, X_batch, Y_batch):
        z = np.dot(X_batch, self.weights)
        return ((z - Y_batch) ** 2).mean()

    def _train_step(self, X_batch, Y_batch, learning_rate, L2):
        z = np.dot(X_batch, self.weights)
        batch_loss = self._loss(X_batch, Y_batch)
        # Add L2 regularization
        gradient = np.dot(X_batch.T, (z - Y_batch)) / Y_batch.size
        gradient += L2 * self.weights
        self.weights -= learning_rate * gradient
        return batch_loss

    def _predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.dot(X, self.weights)

    def predict(self, X):
        return self._predict(X).reshape(-1, 1)

    def score(self, X, Y):
        # Prepare the data
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Y = Y.reshape(-1)
        # Calculate the loss
        loss = self._loss(X, Y)
        return loss

    def train(
        self,
        X,
        Y,
        learning_rate=1e-8,
        epochs=10,
        batch_size=32,
        L2=1e-2,
        method="LstSq",
    ):
        self._build(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Y = Y.reshape(-1)

        if method == "GD":
            for epoch in range(epochs):
                # Shuffle the data
                p = np.random.permutation(X.shape[0])
                X = X[p]
                Y = Y[p]
                pb = tqdm(range(0, X.shape[0], batch_size), dynamic_ncols=True)
                for i in pb:
                    X_batch = X[i : i + batch_size]
                    Y_batch = Y[i : i + batch_size]
                    batch_loss = self._train_step(X_batch, Y_batch, learning_rate, L2)
                    pb.set_description(f"Epoch {epoch + 1}/{epochs}")
                    pb.set_postfix_str(f"Loss: {batch_loss:.4f}")
        elif method == "LstSq":
            # Use the closed form solution with L2 regularization
            self.weights = np.dot(
                np.linalg.pinv(np.dot(X.T, X) + L2 * np.identity(X.shape[1])),
                np.dot(X.T, Y),
            )


# ---(1.1)----------------------------------------------------------------------
# Visualize the data, TSNE
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)
plt.figure(figsize=(8, 8))
sns.set_theme(style="darkgrid")
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y.reshape(-1))
plt.title("TSNE Visualization of the Dataset")
plt.show()

# No scaling, no regularization
model = LinearRegression("NsNr")
model.train(X_train, Y_train, epochs=100, learning_rate=1e-4, L2=0)
print(f"Training Loss | {model.name}: {model.score(X_train, Y_train):.4f}")
print(f"Testing Loss | {model.name}: {model.score(X_test, Y_test):.4f}")

# No scaling, with regularization
model = LinearRegression("NsWr")
model.train(X_train, Y_train, epochs=100, learning_rate=1e-4, L2=1e-2)
print(f"Training Loss | {model.name}: {model.score(X_train, Y_train):.4f}")
print(f"Testing Loss | {model.name}: {model.score(X_test, Y_test):.4f}")

# Experiment with λ value in ridge regression case: plot the test/training error
# Vs ln(λ)
L2 = np.logspace(-5, 5, 21)
train_errors = []
test_errors = []
for l2 in L2:
    model = LinearRegression("NsWr")
    model.train(X_train, Y_train, epochs=10, learning_rate=1e-8, L2=l2)
    train_errors.append(model.score(X_train, Y_train))
    test_errors.append(model.score(X_test, Y_test))

plt.figure(figsize=(8, 8))
sns.set_theme(style="darkgrid")
sns.lineplot(x=np.log(L2), y=train_errors, label="Training Error")
sns.lineplot(x=np.log(L2), y=test_errors, label="Testing Error")
plt.xlabel("ln(λ)")
plt.ylabel("Error")
plt.title("Error Vs ln(λ)")
plt.show()

# ---(1.2)----------------------------------------------------------------------
# TSNE plot after MinMax scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)
plt.figure(figsize=(8, 8))
sns.set_theme(style="darkgrid")
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y.reshape(-1))
plt.title("TSNE Visualization of the Dataset - MinMax Scaled")
plt.show()

# TSNE plot after Standard scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)
plt.figure(figsize=(8, 8))
sns.set_theme(style="darkgrid")
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=Y.reshape(-1))
plt.title("TSNE Visualization of the Dataset - Standard Scaled")
plt.show()

# MinMax scaling, no regularization
model = LinearRegression("MsNr", scaler=MinMaxScaler())
model.train(X_train, Y_train, epochs=100, learning_rate=1e-4, L2=0)
print(f"Training Loss | {model.name}: {model.score(X_train, Y_train):.4f}")
print(f"Testing Loss | {model.name}: {model.score(X_test, Y_test):.4f}")

# MinMax scaling, with regularization
model = LinearRegression("MsWr", scaler=MinMaxScaler())
model.train(X_train, Y_train, epochs=100, learning_rate=1e-4, L2=1e-2)
print(f"Training Loss | {model.name}: {model.score(X_train, Y_train):.4f}")
print(f"Testing Loss | {model.name}: {model.score(X_test, Y_test):.4f}")

# Experiment with λ value in ridge regression case: plot the test/training error
# Vs ln(λ) - for MinMax scaled data
L2 = np.logspace(-5, 5, 21)
train_errors = []
test_errors = []
for l2 in L2:
    model = LinearRegression("MsWr", scaler=MinMaxScaler())
    model.train(X_train, Y_train, epochs=10, learning_rate=1e-8, L2=l2)
    train_errors.append(model.score(X_train, Y_train))
    test_errors.append(model.score(X_test, Y_test))

plt.figure(figsize=(8, 8))
sns.set_theme(style="darkgrid")
sns.lineplot(x=np.log(L2), y=train_errors, label="Training Error")
sns.lineplot(x=np.log(L2), y=test_errors, label="Testing Error")
plt.xlabel("ln(λ)")
plt.ylabel("Error")
plt.title("Error Vs ln(λ) - Data MinMax Scaled")
plt.show()

# Standard scaling, no regularization
model = LinearRegression("SsNr", scaler=StandardScaler())
model.train(X_train, Y_train, epochs=100, learning_rate=1e-4, L2=0)
print(f"Training Loss | {model.name}: {model.score(X_train, Y_train):.4f}")
print(f"Testing Loss | {model.name}: {model.score(X_test, Y_test):.4f}")

# Standard scaling, with regularization
model = LinearRegression("SsWr", scaler=StandardScaler())
model.train(X_train, Y_train, epochs=100, learning_rate=1e-4, L2=1e-2)
print(f"Training Loss | {model.name}: {model.score(X_train, Y_train):.4f}")
print(f"Testing Loss | {model.name}: {model.score(X_test, Y_test):.4f}")

# Experiment with λ value in ridge regression case: plot the test/training error
# Vs ln(λ) - for Standard scaled data
L2 = np.logspace(-5, 5, 21)
train_errors = []
test_errors = []
for l2 in L2:
    model = LinearRegression("SsWr", scaler=StandardScaler())
    model.train(X_train, Y_train, epochs=10, learning_rate=1e-8, L2=l2)
    train_errors.append(model.score(X_train, Y_train))
    test_errors.append(model.score(X_test, Y_test))

plt.figure(figsize=(8, 8))
sns.set_theme(style="darkgrid")
sns.lineplot(x=np.log(L2), y=train_errors, label="Training Error")
sns.lineplot(x=np.log(L2), y=test_errors, label="Testing Error")
plt.xlabel("ln(λ)")
plt.ylabel("Error")
plt.title("Error Vs ln(λ) - Data Standard Scaled")
plt.show()
