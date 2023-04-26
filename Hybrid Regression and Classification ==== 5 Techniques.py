"""This program loads the ADNI dataset as a Pandas dataframe, encodes the Diagnosis variable,
 splits the data into training and testing sets, trains Random Forest, SVM, K-Nearest Neighbors, 
 Logistic Regression, and Gradient Boosting models, combines the probabilities from all models 
 using a simple averaging method, evaluates the hybrid model with the testing set, and plots 
 the ROC curve for the hybrid model with the multiclass target. The program prints out the 
 accuracy, precision, recall, and F1 score for the hybrid model
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load ADNI dataset as a Pandas dataframe
adni_df = pd.read_csv('AD_train.csv')

# Encode the Diagnosis variable
le = LabelEncoder()
adni_df['Diagnosis'] = le.fit_transform(adni_df['Diagnosis'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(adni_df.iloc[:,1:], adni_df['Diagnosis'], test_size=0.2, random_state=42)

# Train with Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Train with SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train with Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100)
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict_proba(X_test)
svm_pred = svm_model.predict_proba(X_test)
knn_pred = knn_model.predict_proba(X_test)
lr_pred = lr_model.predict_proba(X_test)
gb_pred = gb_model.predict_proba(X_test)

# Combine probabilities from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + lr_pred + gb_pred) / 5

# Evaluate hybrid model with testing set
print("Hybrid Classification Metrics:")
print("Accuracy:", accuracy_score(y_test, hybrid_pred.argmax(axis=1)))
print("Precision:", precision_score(y_test, hybrid_pred.argmax(axis=1), average='weighted'))
print("Recall:", recall_score(y_test, hybrid_pred.argmax(axis=1), average='weighted'))
print("F1 Score:", f1_score(y_test, hybrid_pred.argmax(axis=1), average='weighted'))

# Compute and plot ROC curve for hybrid model
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), hybrid_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), hybrid_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(len(le.classes_)):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(le.inverse_transform([i])[0], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Hybrid Model')
plt.legend(loc="lower right")
plt.show()

###########################################################################################
"""This program loads the ADNI dataset as a Pandas dataframe, splits the data into features 
and target, scales the features using StandardScaler, splits the data into training and testing
 sets, trains Random Forest, SVM, K-Nearest Neighbors, Decision Tree, and Gradient Boosting 
 models, combines the predictions from all models using a simple averaging method, evaluates 
 the hybrid model with the testing set, and prints out the Mean Squared Error (MSE), 
 Mean Absolute Error (MAE), and R-squared (R^2) score for the hybrid model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
dataset=pd.read_csv("AD_train.csv")
# Label encoding
le = LabelEncoder() 
dataset['Diagnosis'] = le.fit_transform(dataset['Diagnosis']) 
X=dataset.drop(['Diagnosis'] , axis=1)
y=dataset['Diagnosis']

# Scale features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Train with SVM
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Train with K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train with Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train with Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100)
gb_model.fit(X_train, y_train)

# Combine predictions from all models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Combine predictions from all models
hybrid_pred = (rf_pred + svm_pred + knn_pred + dt_pred + gb_pred) / 5

# Evaluate hybrid model with testing set
print("Hybrid Regression Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, hybrid_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, hybrid_pred))
print("R-squared (R^2) Score:", r2_score(y_test, hybrid_pred))