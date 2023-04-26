import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold


# Load the training dataset
df_train = pd.read_csv('AD_train.csv')

# Define the features and target for training
X_train = df_train.drop(['Status'], axis=1)
y_train = df_train['Status']

# Label encode the target variable for training
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Scale the features using MinMaxScaler for training
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Normalize the features for training
X_train = normalize(X_train)

# Define the models to use
models = [
    XGBRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    LGBMRegressor(),
    NuSVR()
]

# Train each model on the full training set and evaluate using 5-fold cross-validation
for model in models:
    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_pred)
        mae = mean_absolute_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)
        scores.append((mse, mae, r2))
    mse = sum([x[0] for x in scores])/len(scores)
    mae = sum([x[1] for x in scores])/len(scores)
    r2 = sum([x[2] for x in scores])/len(scores)
    print("Model:", type(model).__name__)
    print("Mean Squared Error (MSE): {:.2f}".format(mse))
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("R-Squared: {:.2f}".format(r2))
    print("\n")