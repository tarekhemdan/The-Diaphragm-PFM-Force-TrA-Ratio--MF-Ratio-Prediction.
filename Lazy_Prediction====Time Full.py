# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Define the features and target
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

# Label encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize the features
X_train = normalize(X_train)
X_test = normalize(X_test)

# Initialize LazyRegressor model
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit the model and calculate time consumed
start_time = time.time()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
end_time = time.time()

# Calculate and print the metrics (MSE, MAE, and R2)
for model_name, prediction in predictions.items():
    mse = mean_squared_error(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    print(f"{model_name}: MSE={mse}, MAE={mae}, R2={r2}")

# Print the time consumed
print("Time consumed:", end_time - start_time)