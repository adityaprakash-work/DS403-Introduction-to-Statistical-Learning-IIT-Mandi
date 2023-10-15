# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---DATA-----------------------------------------------------------------------
# Regressor: A 2D dataset X of 1500 samples from a normal distribution N(0, 1)
# Response: Y = βX + ε, where β = [β0, β1, β2]T, β ~ U(1, 2) and ε ~ N(0, 1)
X = np.random.normal(0, 1, (1500, 2))
β = np.random.uniform(1, 2, (3, 1))
ε = np.random.normal(-0.1, 0.1, (1500, 1))
Xb = np.hstack((np.ones((1500, 1)), X))
Y = Xb @ β + ε
Xb_train, Xb_test, Y_train, Y_test = train_test_split(Xb, Y, test_size=0.2)

# Visualize the data in 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], Y, c="r", marker="o")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
plt.show()

# ---MODEL----------------------------------------------------------------------
# SciKit-Learn Linear Regression
M1 = LinearRegression()
M1.fit(Xb_train, Y_train)
# M1 parameters
P1 = M1.coef_.T
P1[0] = M1.intercept_

# Model described in b(ii)
eta = 0.01
theta = np.zeros((3, 1))
iters = 2000


def compCost(X, y, theta):
    tempval = np.dot(X, theta) - y
    return np.sum(np.power(tempval, 2)) / (2 * X.shape[0])


def gradDes(X, y, theta, eta, iters):
    cost = np.zeros(iters)
    for i in range(iters):
        tempval = np.dot(X, theta) - y
        tempval = np.dot(X.T, tempval)
        theta = theta - (eta / X.shape[0]) * tempval
        cost[i] = compCost(X, y, theta)
    return theta, cost


# P2 parameters and cost
P2, cost_vs_iter = gradDes(Xb_train, Y_train, theta, eta, iters)
# Parameter comparison
print("P1: ", P1)
print("P2: ", P2)
# Report the differences between the ground truth β and the obtained β values in
# P1, P2
print("P1 - β: ", P1 - β)
print("P2 - β: ", P2 - β)
# Cost vs. Iteration plot
plt.plot(np.arange(iters), cost_vs_iter)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# ---METRICS--------------------------------------------------------------------
# Use P1, P2 to report Root Mean Square Error, Mean Absolute Error and Normalised
# Root Mean Square Error on the test set.
Y_pred_P1 = Xb_test @ P1
Y_pred_P2 = Xb_test @ P2

# RMSE
RMSE_P1 = np.sqrt(mean_squared_error(Y_test, Y_pred_P1))
RMSE_P2 = np.sqrt(mean_squared_error(Y_test, Y_pred_P2))
# MAE
MAE_P1 = np.mean(np.abs(Y_test - Y_pred_P1))
MAE_P2 = np.mean(np.abs(Y_test - Y_pred_P2))
# NRMSE
NRMSE_P1 = RMSE_P1 / np.std(Y_test)
NRMSE_P2 = RMSE_P2 / np.std(Y_test)

print("RMSE P1: ", RMSE_P1)
print("RMSE P2: ", RMSE_P2)
print("MAE P1: ", MAE_P1)
print("MAE P2: ", MAE_P2)
print("NRMSE P1: ", NRMSE_P1)
print("NRMSE P2: ", NRMSE_P2)

# ---VISUALIZATION--------------------------------------------------------------
# Report a plot with x-axis as test set indices, y-axis as the predicted values
# using P1, P2. Use different colors and markers for the plot.
plt.scatter(np.arange(Y_test.shape[0]), Y_test, c="r", marker="o")
plt.scatter(np.arange(Y_test.shape[0]), Y_pred_P1, c="b", marker="x")
plt.scatter(np.arange(Y_test.shape[0]), Y_pred_P2, c="g", marker="^")
plt.xlabel("Test Set Indices")
plt.ylabel("Predicted Values")
plt.show()

# Report a box-plot of errors across all test set points for the two different
# prediction modes.
errors_P1 = Y_test[:, 0] - Y_pred_P1[:, 0]
errors_P2 = Y_test[:, 0] - Y_pred_P2[:, 0]
plt.boxplot([errors_P1, errors_P2], labels=["P1", "P2"])
plt.xticks([1, 2], ["P1", "P2"])
plt.ylabel("Errors")
plt.show()

# ---END------------------------------------------------------------------------
