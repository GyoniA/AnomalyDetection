from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Load the train and test datasets
start_time = datetime.now()
print("Loading datasets...")
train_path = 'data/pointe77/credit-card-transaction/credit_card_transaction_train.csv'
test_path = 'data/pointe77/credit-card-transaction/credit_card_transaction_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Limit training/testing data to 400,000 rows, to speed up training during development
train_data = train_data.head(800000)
test_data = test_data.head(800000)

# Drop unnecessary columns # TODO: Re-add street, first and last name columns
drop_columns = ['Unnamed: 0', 'cc_num', 'trans_num', 'first', 'last', 'street']
train_data = train_data.drop(columns=drop_columns)
test_data = test_data.drop(columns=drop_columns)

# Process 'trans_date_trans_time' column (convert date to useful numerical features)
# We'll extract month, day, hour minute second from the date to get more usable information
# Format: YYYY-MM-DD HH:MM:SS
def process_dates(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day'] = df['trans_date_trans_time'].dt.day
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['minute'] = df['trans_date_trans_time'].dt.minute
    df['second'] = df['trans_date_trans_time'].dt.second
    df = df.drop(columns=['trans_date_trans_time'])  # Drop the original date column after processing
    return df

# Process 'dob' column (convert date of birth to useful numerical features)
# We'll extract year, month and day from the date to get more usable information
# Format: YYYY-MM-DD
def process_dob(df):
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
    df['dob_year'] = df['dob'].dt.year
    df['dob_month'] = df['dob'].dt.month
    df = df.drop(columns=['dob'])  # Drop the original date of birth column after processing
    return df

train_data = process_dates(train_data)
train_data = process_dob(train_data)

test_data = process_dates(test_data)
test_data = process_dob(test_data)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['merchant', 'category', 'gender', 'city', 'state', 'job']

for column in categorical_columns:
    # Combine train and test columns to ensure consistency of encoding
    combined_data = pd.concat([train_data[column], test_data[column]], axis=0)
    # Convert to categorical type and encode as numerical
    train_data[column] = pd.Categorical(train_data[column], categories=combined_data.unique()).codes
    test_data[column] = pd.Categorical(test_data[column], categories=combined_data.unique()).codes

# Handle missing values (NaNs)
# Impute numerical features with the mean and categorical features with the most frequent value
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Separate numeric and categorical columns
numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
available_categorical_columns = train_data.select_dtypes(include=['category', 'object']).columns.tolist()  # Refresh the list

# Apply imputers to both train and test data if the columns exist
if numeric_columns:
    train_data[numeric_columns] = num_imputer.fit_transform(train_data[numeric_columns])
    test_data[numeric_columns] = num_imputer.transform(test_data[numeric_columns])

if available_categorical_columns:  # Only apply imputation if there are categorical columns
    train_data[available_categorical_columns] = cat_imputer.fit_transform(train_data[available_categorical_columns])
    test_data[available_categorical_columns] = cat_imputer.transform(test_data[available_categorical_columns])

# Separate features and target ('is_fraud' is the target)
X_train = train_data.drop(columns=['is_fraud'])
y_train = train_data['is_fraud']

X_test = test_data.drop(columns=['is_fraud'])
y_test = test_data['is_fraud']

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Data loading and preprocessing completed in {(datetime.now() - start_time).seconds} seconds")

start_time = datetime.now()
# Train and test anomaly detection models
# 1. Isolation Forest
print("\nTraining Isolation Forest...")
isolation_forest = IsolationForest(contamination=0.02)
isolation_forest.fit(X_train_scaled)
print(f"Isolation Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_if = isolation_forest.predict(X_test_scaled)
pred_if = np.where(pred_if == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model
joblib.dump(isolation_forest, 'models/iso_forest.pkl')

start_time = datetime.now()
# 2. Local Outlier Factor
print("\nTraining Local Outlier Factor...")
lof = LocalOutlierFactor(contamination="auto", novelty=True, n_neighbors=10)
lof.fit(X_train_scaled)
print(f"LOF training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_lof = lof.predict(X_test_scaled)
pred_lof = np.where(pred_lof == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model
joblib.dump(lof, 'models/lof.pkl')

start_time = datetime.now()
# 3. One-Class SVM
print("\nTraining One-Class SVM...")
ocsvm = OneClassSVM(nu=0.005, kernel='rbf', gamma='scale')
ocsvm.fit(X_train_scaled)
print(f"OCSVM training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_ocsvm = ocsvm.predict(X_test_scaled)
pred_ocsvm = np.where(pred_ocsvm == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model
joblib.dump(ocsvm, 'models/ocsvm.pkl')

start_time = datetime.now()
# 4. K-Means clustering
print("\nTraining K-Means...")
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_train_scaled)
print(f"K-Means training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_kmeans = kmeans.predict(X_test_scaled)
pred_kmeans = np.where(pred_kmeans == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model
joblib.dump(kmeans, 'models/kmeans.pkl')

print("\n--- Evaluation Results ---\n")

# Isolation Forest evaluation
print("Isolation Forest:")
print(classification_report(y_test, pred_if))
cm_if = ConfusionMatrixDisplay.from_predictions(y_test, pred_if, display_labels=["Non-Fraud", "Fraud"])
plt.title("Isolation Forest Confusion Matrix")

# Local Outlier Factor evaluation
print("Local Outlier Factor:")
print(classification_report(y_test, pred_lof))
cm_lof = ConfusionMatrixDisplay.from_predictions(y_test, pred_lof, display_labels=["Non-Fraud", "Fraud"])
plt.title("LOF Confusion Matrix")

# One-Class SVM evaluation
print("One-Class SVM:")
print(classification_report(y_test, pred_ocsvm))
cm_ocsvm = ConfusionMatrixDisplay.from_predictions(y_test, pred_ocsvm, display_labels=["Non-Fraud", "Fraud"])
plt.title("One-Class SVM Confusion Matrix")

# K-Means evaluation
print("K-Means:")
print(classification_report(y_test, pred_kmeans))
cm_kmeans = ConfusionMatrixDisplay.from_predictions(y_test, pred_kmeans, display_labels=["Non-Fraud", "Fraud"])
plt.title("K-Means Confusion Matrix")

plt.show()

# Function to plot Precision-Recall curves
def plot_precision_recall(y_tests, pred, model_name):
    precision, recall, _ = precision_recall_curve(y_tests, pred)
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')

# ROC Curve
def plot_roc(y_tests, pred, model_name):
    fpr, tpr, _ = roc_curve(y_tests, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')

# Plot precision-recall and ROC for each model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_precision_recall(y_test, pred_if, 'Isolation Forest')
plot_precision_recall(y_test, pred_lof, 'LOF')
plot_precision_recall(y_test, pred_ocsvm, 'One-Class SVM')
plot_precision_recall(y_test, pred_kmeans, 'K-Means')
plt.legend()
plt.subplot(1, 2, 2)
plot_roc(y_test, pred_if, 'Isolation Forest')
plot_roc(y_test, pred_lof, 'LOF')
plot_roc(y_test, pred_ocsvm, 'One-Class SVM')
plot_roc(y_test, pred_kmeans, 'K-Means')
plt.legend()
plt.tight_layout()
plt.show()
