import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import data_loader

X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_pointe77_data()

start_time = datetime.now()
# Train and test anomaly detection models
# 1. Isolation Forest
# Check if the model has already been trained
if os.path.exists('models/iso_forest.pkl'):
    print("\nLoading Isolation Forest from models/iso_forest.pkl...")
    isolation_forest = joblib.load('models/iso_forest.pkl')
    print(f"Isolation Forest loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining Isolation Forest...")
    isolation_forest = IsolationForest(contamination=0.02)
    isolation_forest.fit(X_train_scaled)
    joblib.dump(isolation_forest, 'models/iso_forest.pkl')
    print(f"Isolation Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_if = isolation_forest.predict(X_test_scaled)
pred_if = np.where(pred_if == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model

start_time = datetime.now()
# 2. Local Outlier Factor
# Check if the model has already been trained
if os.path.exists('models/lof.pkl'):
    print("\nLoading Local Outlier Factor from models/lof.pkl...")
    lof = joblib.load('models/lof.pkl')
    print(f"LOF loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination="auto", novelty=True, n_neighbors=10)
    lof.fit(X_train_scaled)
    joblib.dump(lof, 'models/lof.pkl')
    print(f"LOF training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_lof = lof.predict(X_test_scaled)
pred_lof = np.where(pred_lof == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model

start_time = datetime.now()
# 3. One-Class SVM
# Check if the model has already been trained
if os.path.exists('models/ocsvm.pkl'):
    print("\nLoading One-Class SVM from models/ocsvm.pkl...")
    ocsvm = joblib.load('models/ocsvm.pkl')
    print(f"OCSVM loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining One-Class SVM...")
    ocsvm = OneClassSVM(nu=0.005, kernel='rbf', gamma='scale')
    ocsvm.fit(X_train_scaled)
    joblib.dump(ocsvm, 'models/ocsvm.pkl')
    print(f"OCSVM training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_ocsvm = ocsvm.predict(X_test_scaled)
pred_ocsvm = np.where(pred_ocsvm == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model

start_time = datetime.now()
# 4. K-Means clustering
# Check if the model has already been trained
if os.path.exists('models/kmeans.pkl'):
    print("\nLoading K-Means from models/kmeans.pkl...")
    kmeans = joblib.load('models/kmeans.pkl')
    print(f"K-Means loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_train_scaled)
    joblib.dump(kmeans, 'models/kmeans.pkl')
    print(f"K-Means training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_kmeans = kmeans.predict(X_test_scaled)
pred_kmeans = np.where(pred_kmeans == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
# Save the model

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
