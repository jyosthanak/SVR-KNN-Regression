# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'/Users/jyosthanakadiyam/Desktop/Full Stack DS/Nov 2023/9th/9th/emp_sal.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf",degree = 3, gamma = 'scale')
regressor.fit(X, y)
svr_pred = regressor.predict([[6.5]])
svr_pred

regressor = SVR(kernel="rbf",degree = 4, gamma = 'auto')
regressor.fit(X, y)
svr_pred = regressor.predict([[6.5]])
svr_pred

regressor = SVR(kernel="poly",degree = 4, gamma = 'scale')
regressor.fit(X, y)
svr_pred = regressor.predict([[6.5]])
svr_pred

regressor = SVR(kernel="poly",degree = 5, gamma = 'scale')
regressor.fit(X, y)
svr_pred = regressor.predict([[6.5]])
svr_pred

regressor = SVR(kernel="poly",degree = 5 gamma = 'auto')
regressor.fit(X, y)
svr_pred = regressor.predict([[6.5]])
svr_pred
