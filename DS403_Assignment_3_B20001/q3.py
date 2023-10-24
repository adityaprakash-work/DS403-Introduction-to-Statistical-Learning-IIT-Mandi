# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

# ---DATASET--------------------------------------------------------------------
# --Description: Apple stock data
# Columns: Date, Close/Last, Volume, Open, High, Low)
# Date: Date of the trading day (MM/DD/YYYY)
# Close/Last: Closing price of the stock for the day (in USD)
# Volume: Number of shares traded on the day (in millions)
# Open: Opening price of the stock for the day (in USD)
# High: Highest price of the stock for the day (in USD)
# Low: Lowest price of the stock for the day (in USD)

# Data preprocessing
data = pd.read_excel("Regression_datasets/Apple_stock_data.xlsx")
print(data.head())
# Converting columns to proper format
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(by="Date")
data["Close/Last"] = data["Close/Last"].str.replace("$", "").astype(float)
data["Volume"] = data["Volume"].astype(float)
data["Open"] = data["Open"].str.replace("$", "").astype(float)
data["High"] = data["High"].str.replace("$", "").astype(float)
data["Low"] = data["Low"].str.replace("$", "").astype(float)

# Plot: Close/Last vs Date
plt.figure(figsize=(16, 8))
sns.set_style("darkgrid")
sns.lineplot(x="Date", y="Close/Last", data=data, color="green")
plt.title("Apple stock data - Close/Last vs Date")
plt.xlabel("Date")
plt.ylabel("Close/Last")
plt.show()

# Dataset split
Xtr = data[data["Date"] < "2022-01-01"]
Xte = data[data["Date"] >= "2022-01-01"]


# ---MODEL----------------------------------------------------------------------
# --Description: Multivariate Polynomial Regression model
class TemporalPolynomialRegressor:
    def __init__(self, name, n_ivars=None, degree=2, window_size=10):
        self.name = name
        self.n_ivars = n_ivars
        self.degree = degree
        self.window_size = window_size
        self.weights = None
        if n_ivars is not None:
            self._build()
        self.X_preprocessor = StandardScaler()
        self.Y_preprocessor = StandardScaler()

    def _build(self):
        self.weights = np.zeros(self.n_ivars * self.window_size * (self.degree + 1))

    def gen_tpr_data(self, X, Y=None, time_steps=None, update_preprocessor=False):
        if time_steps is None:
            time_steps = np.arange(self.window_size, X.shape[0])
        X_tprd = []
        Y_tprd = []
        for time_step in time_steps:
            if time_step < self.window_size or time_step >= X.shape[0]:
                continue
            X_time_slice = X[time_step - self.window_size : time_step]
            X_time_slice = np.array(
                [X_time_slice**i for i in range(self.degree + 1)]
            ).flatten()
            X_tprd.append(X_time_slice)
            if Y is not None:
                target = Y[time_step]
                Y_tprd.append(target)
        X_tprd = np.array(X_tprd)
        Y_tprd = np.array(Y_tprd)
        if update_preprocessor:
            self.X_preprocessor.partial_fit(X_tprd)
            self.Y_preprocessor.partial_fit(Y_tprd.reshape(-1, 1))
        X_tprd = self.X_preprocessor.transform(X_tprd)
        Y_tprd = self.Y_preprocessor.transform(Y_tprd.reshape(-1, 1)).reshape(-1)
        return X_tprd, Y_tprd

    def train(
        self,
        X,
        Y,
        time_steps=None,
        method="gradient_descent",
        learning_rate=1e-3,
        batch_size=32,
        epochs=1000,
        priming_epochs=3,
        L1=1e-7,
        L2=1e-7,
    ):
        if self.weights is None:
            self.n_ivars = X.shape[1]
            self._build()
        if time_steps is None:
            time_steps = np.arange(self.window_size, X.shape[0])
            np.random.shuffle(time_steps)

        epoch_loss_history = []

        if method == "gradient_descent":
            for epoch in range(epochs):
                if epoch < priming_epochs:
                    update_preprocessor = True
                else:
                    update_preprocessor = False
                np.random.shuffle(time_steps)
                batch_losses = []
                progress_bar = tqdm(
                    range(0, len(time_steps), batch_size),
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                )
                for i in progress_bar:
                    batch_time_steps = time_steps[i : i + batch_size]
                    X_tprd, Y_tprd = self.gen_tpr_data(
                        X, Y, batch_time_steps, update_preprocessor
                    )
                    Y_pred = X_tprd @ self.weights
                    # Adding L1 and L2 regularization
                    loss = np.mean((Y_pred - Y_tprd) ** 2)
                    batch_losses.append(loss)
                    l1_reg = L1 * np.sign(self.weights)
                    l2_reg = L2 * self.weights
                    grad = X_tprd.T @ (Y_pred - Y_tprd) + l1_reg + l2_reg

                    self.weights -= learning_rate * grad / batch_size

                    avg_loss = np.mean(batch_losses)
                    progress_bar.set_description(
                        f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.6f}"
                    )

                mean_epoch_loss = np.mean(batch_losses)
                epoch_loss_history.append(mean_epoch_loss)

        elif method == "least_squares":
            X_tprd, Y_tprd = self.gen_tpr_data(X, Y, time_steps, True)
            X_tprd = self.X_preprocessor.transform(X_tprd)
            Y_tprd = self.Y_preprocessor.transform(Y_tprd.reshape(-1, 1)).reshape(-1)
            self.weights = np.linalg.pinv(X_tprd.T @ X_tprd) @ (X_tprd.T @ Y_tprd)

        else:
            raise ValueError("Invalid method")

        return epoch_loss_history

    def predict(self, X_tprd, Y_tprd=None):
        Y_pred = X_tprd @ self.weights
        accuracy = None
        if Y_tprd is not None:
            accuracy = np.mean((Y_pred - Y_tprd) ** 2)
        Y_pred = self.Y_preprocessor.inverse_transform(Y_pred.reshape(-1, 1)).reshape(
            -1
        )
        return Y_pred, accuracy

    def plot_weights(self, feature_indices=None):
        # Take a list of feature indices to plot, for those features plot the
        # weights as 2D, degree + 1 x window_size images, on the same figure.
        # subplots are feature-wise
        weights = self.weights.reshape(self.degree + 1, self.window_size, self.n_ivars)
        if feature_indices is None:
            feature_indices = np.arange(self.n_ivars)

        for feature_index in feature_indices:
            fig, axes = plt.subplots()
            heatmap = axes.imshow(
                weights[:, :, feature_index], cmap="viridis"
            )  # Use 'viridis' colormap for heatmap
            axes.set_title(f"Feature {feature_index}")
            axes.set_xlabel("Window (t - w to t)")
            axes.set_ylabel("Degree")

            # Invert the y-axis ticks and label each pixel with an integer
            axes.invert_yaxis()
            axes.set_yticks(np.arange(self.degree + 1))
            axes.set_xticks(np.arange(self.window_size))
            axes.set_yticklabels(np.arange(self.degree, -1, -1))
            axes.set_xticklabels(np.arange(1, self.window_size + 1))
            for i in range(self.degree + 1):
                for j in range(self.window_size):
                    axes.text(
                        j,
                        i,
                        f"{weights[i, j, feature_index]:.2f}",
                        ha="center",
                        va="center",
                        color="w",
                    )

            plt.suptitle(f"{self.name}")
            plt.colorbar(heatmap)  # Add colorbar to the heatmap
            axes.grid(False)  # Remove grid
            plt.show()


# ---TRAINING-------------------------------------------------------------------
# --Description: Training the model on the Close/Last column of the dataset,
# i.e predicting the Close/Last value for the next day.
# Data
X = np.array(Xtr["Close/Last"])
Xn = X.reshape(-1, 1)
Yn = X
Xtr = Xn[: int(0.8 * Xn.shape[0])]
Ytr = Yn[: int(0.8 * Yn.shape[0])]
Xte = Xn[int(0.8 * Xn.shape[0]) :]
Yte = Yn[int(0.8 * Yn.shape[0]) :]

# Create models for degrees 1 to 5 and window sizes 1 to 20
models = []
for degree in range(1, 6):
    for window_size in range(1, 6):
        model = TemporalPolynomialRegressor(
            name=f"Model_D{degree}W{window_size}",
            n_ivars=1,
            degree=degree,
            window_size=window_size,
        )
        models.append(model)

# Training models
for model in models:
    print(f"Model d = {model.degree} and w = {model.window_size}")
    model.train(Xtr, Ytr, method="gradient_descent", epochs=2, batch_size=32)
    print("Training complete\n")

# ---EVALUATION-----------------------------------------------------------------
# --Description: Evaluating the models on the test data# Plotting:
# i) Plot the prediction error Vs D (order of the polynomial fit - model complexity)
# ii) Plot the prediction error Vs W
# iii) 3-D plot Test error (prediction loss) Vs D and W.
# iv) Plot predictions vs original on same graph

# i
dataD = []
# ii
dataW = []
# iii
data3d = []
# iv
plt.figure(figsize=(16, 8))
sns.set_style("darkgrid")
sns.lineplot(Yte.flatten(), label="Original")

for model in models:
    print(f"Model d = {model.degree} and w = {model.window_size}")
    # Generate test data
    Xte_tprd, Yte_tprd = model.gen_tpr_data(Xte, Yte)
    # Predict
    Yte_pred, loss = model.predict(Xte_tprd, Yte_tprd)
    # pad Yte
    Yte_pred = np.pad(
        Yte_pred, (model.window_size, 0), mode="constant", constant_values=np.nan
    )
    print(f"Test Loss: {loss:.6f}")

    # iv
    sns.lineplot(Yte_pred, label=model.name)
    # iii
    data3d.append([model.degree, model.window_size, loss])
    # ii
    dataW.append([model.window_size, loss])
    # i
    dataD.append([model.degree, loss])

plt.title("Test - Apple stock data - Close/Last vs Date")
plt.xlabel("Date")
plt.ylabel("Close/Last")
plt.legend()
plt.show()

# (i)
dataD = np.array(dataD)
# Mean by degree
dataD = np.array(
    [
        [degree, np.mean(dataD[dataD[:, 0] == degree, 1])]
        for degree in np.unique(dataD[:, 0])
    ]
)
plt.figure(figsize=(16, 8))
sns.set_style("darkgrid")
sns.lineplot(dataD[:, 1])
# change ticks with dataD[:, 0]
plt.xticks(np.arange(0, dataD.shape[0]), dataD[:, 0])
plt.title("Prediction error Vs D")
plt.xlabel("D")
plt.ylabel("Prediction error")
plt.show()

# (ii)
dataW = np.array(dataW)
# Mean by window size
dataW = np.array(
    [
        [window_size, np.mean(dataW[dataW[:, 0] == window_size, 1])]
        for window_size in np.unique(dataW[:, 0])
    ]
)
plt.figure(figsize=(16, 8))
sns.set_style("darkgrid")
sns.lineplot(dataW[:, 1])
plt.xticks(np.arange(0, dataW.shape[0]), dataW[:, 0])
plt.title("Prediction error Vs W")
plt.xlabel("W")
plt.ylabel("Prediction error")
plt.show()

# (iii)
data3d = np.array(data3d)
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2])
ax.set_xlabel("D")
ax.set_ylabel("W")
ax.set_zlabel("Prediction error")
plt.title("Test error (prediction loss) Vs D and W")
plt.show()

# ---ANALYSIS-------------------------------------------------------------------
# Analysis of weights
# Plotting the weights for all models
for model in models:
    print(f"Model d = {model.degree} and w = {model.window_size}")
    model.plot_weights()

# ---CONCLUSION-----------------------------------------------------------------
# The best models are usually W = 5 and D = 3

# The less complex models in terms of D have a high bias and as W is increased
# the loss decreases and only increases after a certain point.

# As the degree is increased it takes less and less value of W to overfit the
# data. This is because the model is able to fit the data better with higher
# degree polynomials and hence the loss decreases. But after a certain point
# the model starts to overfit the data and the loss increases.
