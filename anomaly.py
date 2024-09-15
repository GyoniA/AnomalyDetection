from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the train and test datasets
start_time = datetime.now()
print("Loading datasets...")
train_path = 'data/pointe77/credit-card-transaction/credit_card_transaction_train.csv'
test_path = 'data/pointe77/credit-card-transaction/credit_card_transaction_test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Drop unnecessary columns
drop_columns = ['Unnamed: 0', 'cc_num', 'trans_num', 'first', 'last', 'street'] # TODO: Re- street, first and last name columns
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
# We'll extract year and month from the date of birth to get more usable information
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
pred_if = isolation_forest.predict(X_test_scaled)
pred_if = np.where(pred_if == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

# 2. Local Outlier Factor
print("\nTraining Local Outlier Factor...")
lof = LocalOutlierFactor(contamination=0.02, novelty=True)
lof.fit(X_train_scaled)
pred_lof = lof.predict(X_test_scaled)
pred_lof = np.where(pred_lof == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

# 3. One-Class SVM
print("\nTraining One-Class SVM...")
ocsvm = OneClassSVM(nu=0.02, kernel='rbf', gamma='scale')
ocsvm.fit(X_train_scaled)
pred_ocsvm = ocsvm.predict(X_test_scaled)
pred_ocsvm = np.where(pred_ocsvm == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

print(f"Training and testing completed in {(datetime.now() - start_time).seconds} seconds")

# Evaluate the models on the test set
print("\n--- Evaluation Results ---\n")

# Isolation Forest evaluation
print("Isolation Forest:")
print(classification_report(y_test, pred_if))

# Local Outlier Factor evaluation
print("Local Outlier Factor:")
print(classification_report(y_test, pred_lof))

# One-Class SVM evaluation
print("One-Class SVM:")
print(classification_report(y_test, pred_ocsvm))

# Visualize the predictions from each model
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.hist(pred_if, bins=2)
plt.title("Isolation Forest Predictions")

plt.subplot(1, 3, 2)
plt.hist(pred_lof, bins=2)
plt.title("LOF Predictions")

plt.subplot(1, 3, 3)
plt.hist(pred_ocsvm, bins=2)
plt.title("One-Class SVM Predictions")

plt.tight_layout()
plt.show()
