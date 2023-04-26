"""This program loads the ADNI dataset as a Pandas dataframe, encodes the Diagnosis variable,
 normalizes the feature variables, splits the data into training and testing sets, 
 trains an SVM model and a Random Forest model, combines the predictions from both models
 using a simple averaging method, and evaluates the hybrid model with the testing set. 
 The program prints out the mean absolute error, mean squared error, and R^2 score for 
 the hybrid model. Note that you can replace the ADNI dataset with any regression dataset
 of your choice, and also try different combinations of regression models for the hybrid model.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load ADNI dataset as a Pandas dataframe
adni_df = pd.read_csv('AD_train.csv')

# Encode the Diagnosis variable
le = LabelEncoder()
adni_df['Diagnosis'] = le.fit_transform(adni_df['Diagnosis'])

# Normalize the feature variables
scaler = MinMaxScaler()
adni_df.iloc[:,1:] = scaler.fit_transform(adni_df.iloc[:,1:])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(adni_df.iloc[:,1:], adni_df['Diagnosis'], test_size=0.2, random_state=42)

# Train with SVM
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Train with Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Combine predictions from SVM and Random Forest
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
hybrid_pred = (svm_pred + rf_pred) / 2

# Evaluate hybrid model with testing set
print("Hybrid Regression Metrics SVM - Random Forest :")
print("Mean Absolute Error:", mean_absolute_error(y_test, hybrid_pred))
print("Mean Squared Error:", mean_squared_error(y_test, hybrid_pred))
print("R^2 Score:", r2_score(y_test, hybrid_pred))