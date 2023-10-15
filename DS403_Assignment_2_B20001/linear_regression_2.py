# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ---DATA-----------------------------------------------------------------------
# Description:
#   The data set contains 9 variables:
#   lcavol: log cancer volume
#   lweight: log prostate weight
#   age: patient age
#   lbph: log of the amount of benign prostatic hyperplasia
#   svi: seminal vesicle invasion
#   lcp: log of capsular penetration
#   gleason: Gleason score
#   pgg45: percent of Gleason scores 4 or 5
#   lpsa: log prostate specific antigen
#   train: an indicator variable for those observations that were used to
#   construct the models
DATA_URL = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
data = pd.read_csv(DATA_URL, delimiter="\t")
data.drop(["Unnamed: 0"], axis=1, inplace=True)
data_tr = data[data["train"] == "T"].drop(["train"], axis=1)
data_te = data[data["train"] == "F"].drop(["train"], axis=1)

# Standardize the data
scaler = StandardScaler()
scaler.fit_transform(data_tr)

data_tr_std = scaler.transform(data_tr)
data_te_std = scaler.transform(data_te)

# ---MODEL----------------------------------------------------------------------
model = LinearRegression()
model.fit(data_tr_std[:, :-1], data_tr_std[:, -1])
# Model parameters
P1 = np.array([model.intercept_, *model.coef_])
# Model parameter comparison
print("P1: ", P1)
print("P1 (book): [2.46, 0.68, 0.26, -0.14, 0.21, 0.31, -0.29, -0.02, 0.27]")


# ---EVALUATION-----------------------------------------------------------------
# Mean squared error
true_lpsa_te = data_te_std[:, -1]
pred_lpsa_te = model.predict(data_te_std[:, :-1])
MSE = np.mean(np.power(true_lpsa_te - pred_lpsa_te, 2))
print("MSE: ", MSE)
# Standard error
# Var(βhat) = σ^2(X^TX)^-1
# σ^2 = 1/(n-p-1) * Σ(yi - yhat)^2
sigma_sq = np.sum(np.power(true_lpsa_te - pred_lpsa_te, 2)) * (
    1 / (data_tr_std.shape[0] - 8 - 1)
)
XTX_inv = np.linalg.inv(np.dot(data_tr_std[:, :-1].T, data_tr_std[:, :-1]))
SE_beta_hat_test = np.sqrt(sigma_sq * np.diag(XTX_inv))
print("SE_beta_hat_coeff_test: ", SE_beta_hat_test)
# Coefficient of determination (R^2)
# R^2 = 1 - (RSS/TSS)
# RSS = Σ(yi - yhat)^2
# TSS = Σ(yi - ybar)^2
RSS = np.sum(np.power(true_lpsa_te - pred_lpsa_te, 2))
TSS = np.sum(np.power(true_lpsa_te - np.mean(true_lpsa_te), 2))
R_sq = 1 - (RSS / TSS)
print("R_sq: ", R_sq)

# ---END------------------------------------------------------------------------
