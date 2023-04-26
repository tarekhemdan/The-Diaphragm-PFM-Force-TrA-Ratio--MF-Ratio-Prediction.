from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")


# Add any other classifiers you want to try here

# Load data
dataset=pd.read_csv("Data_Fix_Final.csv")
X=dataset.drop(['lumbar angle'] , axis=1)
y=dataset['lumbar angle']
print (X)
print(y)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the 20 algorithms to use
algorithms = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    SVR(kernel='linear'),
    SVR(kernel='rbf'),
    SVR(kernel='poly'),
    KNeighborsRegressor(),
    MLPRegressor(),
    XGBRegressor(),
    LinearRegression(normalize=True),
    Ridge(alpha=0.1),
    DecisionTreeRegressor(max_depth=3),
    RandomForestRegressor(n_estimators=100, max_depth=5),
    GradientBoostingRegressor(n_estimators=100, max_depth=3),
    AdaBoostRegressor(n_estimators=100),
    XGBRegressor(n_estimators=100, max_depth=3)
]

# Define the number of folds for cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform k-fold cross-validation and compute the R2 and MSE metrics for each algorithm
for algorithm in algorithms:
    r2_scores = cross_val_score(algorithm, X, y, cv=kf, scoring='r2')
    mse_scores = cross_val_score(algorithm, X, y, cv=kf, scoring='neg_mean_squared_error')
    print(type(algorithm).__name__)
    print("R2 scores: ", r2_scores)
    print("R2 mean: ", np.mean(r2_scores))
    print("MSE scores: ", -mse_scores)
    print("MSE mean: ", np.mean(-mse_scores))
    print("-------------------------------------------------------")
