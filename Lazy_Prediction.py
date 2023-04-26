import lazypredict

from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import lazypredict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder 
import warnings
warnings.filterwarnings("ignore")


# Load data
dataset=pd.read_csv("AD_train.csv")

# Label encoding
le = LabelEncoder() 
dataset['Diagnosis'] = le.fit_transform(dataset['Diagnosis']) 
X=dataset.drop(['Diagnosis'] , axis=1)
y=dataset['Diagnosis']
print (X)
print(y)


# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Compute pairwise correlations between features
correlations = X.corr()
# Train a random forest classifier to estimate feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Print results
print("Pairwise feature correlations:")
print(correlations)
print("\nFeature importances:")
print(importances)

#X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

