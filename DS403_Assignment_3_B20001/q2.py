# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


# ---DATASET--------------------------------------------------------------------
df = pd.read_csv("Regression_datasets/banknote_authentication.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1:].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# ---MODEL----------------------------------------------------------------------
class LogisticRegression:
    def __init__(self, name):
        self.name = name
        self.scaler = StandardScaler()
        self.weights = None

    def _build(self, X):
        self.weights = np.zeros((X.shape[1] + 1))
        self.scaler.fit(X)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, X_batch, Y_batch):
        z = np.dot(X_batch, self.weights)
        h = self._sigmoid(z)
        return (-Y_batch * np.log(h) - (1 - Y_batch) * np.log(1 - h)).mean()

    def _train_step(self, X_batch, Y_batch, learning_rate, L1=1e-3):
        z = np.dot(X_batch, self.weights)
        h = self._sigmoid(z)
        batch_loss = self._loss(X_batch, Y_batch)
        gradient = np.dot(X_batch.T, (h - Y_batch)) / Y_batch.size
        # Add L1 regularization
        gradient += L1 * np.sign(self.weights)
        self.weights -= learning_rate * gradient
        return batch_loss

    def train(self, X, Y, learning_rate=1e-3, epochs=10, batch_size=32, L1=1e-3):
        self._build(X)
        X = self.scaler.transform(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        Y = Y.reshape(-1)

        for epoch in range(epochs):
            # Shuffle the data
            p = np.random.permutation(X.shape[0])
            X = X[p]
            Y = Y[p]
            pb = tqdm(range(0, X.shape[0], batch_size), dynamic_ncols=True)
            for i in pb:
                X_batch = X[i : i + batch_size]
                Y_batch = Y[i : i + batch_size]
                batch_loss = self._train_step(X_batch, Y_batch, learning_rate, L1)
                pb.set_description(
                    f"Epoch: {epoch + 1} | Training Loss: {batch_loss:.4f}"
                )

    def predict(self, X, threshold=None):
        X = self.scaler.transform(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        z = np.dot(X, self.weights)
        h = self._sigmoid(z)
        if threshold:
            return (h > threshold).astype(int)
        return h

    def evaluate(self, X, Y, threshold=0.5):
        Y_pred = self.predict(X, threshold)
        Y_pred = Y_pred.reshape(-1)
        Y = Y.reshape(-1)
        accuracy = (Y_pred == Y).mean()
        return accuracy


# ---TRAINING-------------------------------------------------------------------
model = LogisticRegression("Logistic Regression")
model.train(X_train, Y_train, learning_rate=1e-2, epochs=100, batch_size=4)

# ---EVALUATION-----------------------------------------------------------------
train_accuracy = model.evaluate(X_train, Y_train)
test_accuracy = model.evaluate(X_test, Y_test)
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# ---CONCLUSION-----------------------------------------------------------------
print(f"Test Accuracy > 0.98: {test_accuracy > 0.98}")

# ---END------------------------------------------------------------------------
