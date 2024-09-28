import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import data_loader
import auto_encoder as ae

X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_pointe77_data()

train_loader, test_loader = data_loader.create_dataloader(X_train_scaled, X_test_scaled)

start_time = datetime.now()
# Train and test anomaly detection models
# 1. Isolation Forest
# Check if the model has already been trained
ISOFOREST_model_path = 'models/iso_forest.pkl'
if os.path.exists(ISOFOREST_model_path):
    print(f"\nLoading Isolation Forest from {ISOFOREST_model_path}...")
    isolation_forest = joblib.load(ISOFOREST_model_path)
    print(f"Isolation Forest loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining Isolation Forest...")
    isolation_forest = IsolationForest(contamination=0.02)
    isolation_forest.fit(X_train_scaled)
    # Save the model
    joblib.dump(isolation_forest, ISOFOREST_model_path)
    print(f"Isolation Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_if = isolation_forest.predict(X_test_scaled)
pred_if = np.where(pred_if == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

start_time = datetime.now()
# 2. Local Outlier Factor
# Check if the model has already been trained
LOF_model_path = 'models/lof.pkl'
if os.path.exists(LOF_model_path):
    print(f"\nLoading Local Outlier Factor from {LOF_model_path}...")
    lof = joblib.load(LOF_model_path)
    print(f"LOF loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination="auto", novelty=True, n_neighbors=10)
    lof.fit(X_train_scaled)
    # Save the model
    joblib.dump(lof, LOF_model_path)
    print(f"LOF training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_lof = lof.predict(X_test_scaled)
pred_lof = np.where(pred_lof == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

start_time = datetime.now()
# 3. One-Class SVM
# Check if the model has already been trained
OCSVM_model_path = 'models/ocsvm.pkl'
if os.path.exists(OCSVM_model_path):
    print(f"\nLoading One-Class SVM from {OCSVM_model_path}...")
    ocsvm = joblib.load(OCSVM_model_path)
    print(f"OCSVM loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining One-Class SVM...")
    ocsvm = OneClassSVM(nu=0.005, kernel='rbf', gamma='scale')
    ocsvm.fit(X_train_scaled)
    # Save the model
    joblib.dump(ocsvm, OCSVM_model_path)
    print(f"OCSVM training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_ocsvm = ocsvm.predict(X_test_scaled)
pred_ocsvm = np.where(pred_ocsvm == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

start_time = datetime.now()
# 4. K-Means clustering
# Check if the model has already been trained
KMEANS_model_path = 'models/kmeans.pkl'
if os.path.exists(KMEANS_model_path):
    print(f"\nLoading K-Means from {KMEANS_model_path}...")
    kmeans = joblib.load(KMEANS_model_path)
    print(f"K-Means loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    print("\nTraining K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_train_scaled)
    # Save the model
    joblib.dump(kmeans, KMEANS_model_path)
    print(f"K-Means training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
pred_kmeans = kmeans.predict(X_test_scaled)
pred_kmeans = np.where(pred_kmeans == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

start_time = datetime.now()
# 5. Autoencoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the autoencoder model
autoencoder_model = ae.Autoencoder(input_dim=X_train_scaled.shape[1])
ae_model_path = 'models/autoencoder.pth'

# Load the trained model if it exists
if os.path.exists(ae_model_path):
    print(f"\nLoading Autoencoder from {ae_model_path}...")
    autoencoder_model.load_state_dict(torch.load(ae_model_path, map_location=device))
    print(f"Autoencoder model loaded in {(datetime.now() - start_time).seconds} seconds")
else:
    # Train the model if it's not already saved
    print("\nTraining Autoencoder...")
    autoencoder_model = ae.train_autoencoder(autoencoder_model, train_loader, device, epochs=10, lr=0.001)
    torch.save(autoencoder_model.state_dict(), ae_model_path)
    print(f"Autoencoder model training completed in {(datetime.now() - start_time).seconds} seconds, weights saved to {ae_model_path}")

# Get reconstruction error on the test data
reconstruction_error = ae.get_reconstruction_error(autoencoder_model, test_loader, device)
# Set a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 98)
# Get predictions based on the threshold
pred_ae = np.where(reconstruction_error > threshold, 1, 0)

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


# Autoencoder evaluation
print("Autoencoder (AE):")
print(classification_report(y_test, pred_ae))
cm_ae = ConfusionMatrixDisplay.from_predictions(y_test, pred_ae, display_labels=["Non-Fraud", "Fraud"])
plt.title("Autoencoder Confusion Matrix")

plt.show()

def plot_precision_recall(y_tests, pred, model_name):
    """
    Plot Precision-Recall curves and compute AUPRC.
    """
    precision, recall, _ = precision_recall_curve(y_tests, pred)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, marker='.', label=f'{model_name} AUPRC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

def plot_roc(y_tests, pred, model_name):
    """
    Plot ROC curves
    """
    fpr, tpr, _ = roc_curve(y_tests, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')

# Plot precision-recall and ROC for each model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_precision_recall(y_test, pred_if, 'Isolation Forest')
plot_precision_recall(y_test, pred_lof, 'LOF')
plot_precision_recall(y_test, pred_ocsvm, 'One-Class SVM')
plot_precision_recall(y_test, pred_kmeans, 'K-Means')
plot_precision_recall(y_test, pred_ae, 'Autoencoder')
plt.legend()
plt.subplot(1, 2, 2)
plot_roc(y_test, pred_if, 'Isolation Forest')
plot_roc(y_test, pred_lof, 'LOF')
plot_roc(y_test, pred_ocsvm, 'One-Class SVM')
plot_roc(y_test, pred_kmeans, 'K-Means')
plot_roc(y_test, pred_ae, 'Autoencoder')
plt.legend()
plt.tight_layout()
plt.show()
