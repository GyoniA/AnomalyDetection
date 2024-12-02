import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch import nn
import auto_encoder as ae

import data_loader
from anomaly import plot_precision_recall, plot_roc
from transformer import compute_class_weights, train_loop, TransformerClassifier, get_optimal_threshold

model_name = 'bot-detection' # TODO: Move this to a config file

model_path = f'models/{model_name}/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test, le_country, le_query = data_loader.load_bot_detection_data()
    # convert xs to numpy array
    x_train_numpy = X_train_scaled.to_numpy()
    x_test_numpy = X_test_scaled.to_numpy()
    train_loader, test_loader = data_loader.create_dataloaders(x_train_numpy, x_test_numpy, use_gpu=torch.cuda.is_available())
    class_train_loader, class_test_loader = data_loader.create_classification_dataloaders(x_train_numpy, x_test_numpy, y_train, y_test)

    start_time = datetime.now()
    # Train and test anomaly detection models
    # 1. Random Forest
    # Check if the model has already been trained
    RF_model_path = model_path + 'rf.pkl'
    if os.path.exists(RF_model_path):
        print(f"\nLoading Random Forest from {RF_model_path}...")
        rf = joblib.load(RF_model_path)
        print(f"Random Forest loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train_scaled, y_train)
        # Save the model
        joblib.dump(rf, RF_model_path)
        print(f"Random Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_rf = rf.predict(X_test_scaled)

    start_time = datetime.now()
    # 2. XGBoost
    # Check if the model has already been trained
    XGB_model_path = model_path + 'xgb.pkl'
    if os.path.exists(XGB_model_path):
        print(f"\nLoading XGBoost from {XGB_model_path}...")
        xgb = joblib.load(XGB_model_path)
        print(f"XGBoost loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining XGBoost...")
        xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train_scaled, y_train)
        # Save the model
        joblib.dump(xgb, XGB_model_path)
        print(f"XGBoost training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_xgb = xgb.predict(X_test_scaled)

    start_time = datetime.now()
    # 3. K-Nearest Neighbors TODO: This model is too slow to predict with the current data
    # KNN_model_path = model_path + 'knn.pkl'
    # if os.path.exists(KNN_model_path):
    #     print(f"\nLoading K-NN from {KNN_model_path}...")
    #     knn = joblib.load(KNN_model_path)
    #     print(f"K-NN loaded in {(datetime.now() - start_time).seconds} seconds")
    # else:
    #     print("\nTraining K-NN model...")
    #     knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski',
    #                                p=2, algorithm='ball_tree', leaf_size=20)
    #     knn.fit(X_train_scaled, y_train)
    #     joblib.dump(knn, KNN_model_path)
    #     print(f"K-NN model training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    # start_time = datetime.now()
    # pred_knn = knn.predict(X_test_scaled)
    # print(f"K-NN prediction completed in {(datetime.now() - start_time).seconds} seconds")
    # 4. Transformer
    # Define the Transformer model
    transformer_model = TransformerClassifier(input_dim=X_train_scaled.shape[1])
    transformer_model = transformer_model.to(device)
    transformer_model_path = model_path + 'transformer.pth'
    if os.path.exists(transformer_model_path):
        print(f"\nLoading Transformer model from {transformer_model_path}...")
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
        print(f"Transformer model loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        # Train the model if it's not already saved
        print("\nTraining Transformer...")
        class_weights = compute_class_weights(y_train)
        class_weights = class_weights.to(device)
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loop(transformer_model, class_train_loader, class_test_loader, criterion, transformer_model_path, device, epochs=5)
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
        print(f"Transformer model training completed in {(datetime.now() - start_time).seconds} seconds, weights saved to {transformer_model_path}")

    # Transformer evaluation
    transformer_model.eval()
    all_labels_transformer = []
    all_probs_transformer = []

    with torch.no_grad():
        for inputs, labels in class_test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = transformer_model(inputs)
            probs = torch.sigmoid(outputs)

            all_labels_transformer.extend(labels.cpu().numpy())
            all_probs_transformer.extend(probs.cpu().numpy())

    # Determine optimal threshold
    optimal_threshold, best_f1 = get_optimal_threshold(np.array(all_labels_transformer), np.array(all_probs_transformer))
    preds_transformer = (np.array(all_probs_transformer) >= optimal_threshold).astype(int)

    contamination = 0.2 if model_name == 'cic-unsw-nb15' else y_train.sum() / len(y_train)
    print(f"Contamination: {contamination}")

    start_time = datetime.now()
    # Train and test anomaly detection models
    # 5. Isolation Forest
    # Check if the model has already been trained
    ISOFOREST_model_path = model_path + 'iso_forest.pkl'
    if os.path.exists(ISOFOREST_model_path):
        print(f"\nLoading Isolation Forest from {ISOFOREST_model_path}...")
        isolation_forest = joblib.load(ISOFOREST_model_path)
        print(f"Isolation Forest loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining Isolation Forest...")
        isolation_forest = IsolationForest(contamination=contamination)
        isolation_forest.fit(X_train_scaled)
        # Save the model
        joblib.dump(isolation_forest, ISOFOREST_model_path)
        print(f"Isolation Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_if = isolation_forest.predict(X_test_scaled)
    pred_if = np.where(pred_if == 1, np.array(0, dtype=pred_if.dtype),
                       np.array(1, dtype=pred_if.dtype))  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    start_time = datetime.now()
    # 6. K-Means clustering
    # Check if the model has already been trained
    KMEANS_model_path = model_path + 'kmeans.pkl'
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
    pred_kmeans = np.where(pred_kmeans == 1, np.array(0, dtype=pred_if.dtype),
                           np.array(1, dtype=pred_if.dtype))  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    start_time = datetime.now()
    # 7. Autoencoder
    # Define the autoencoder model
    autoencoder_model = ae.Autoencoder(input_dim=X_train_scaled.shape[1])
    ae_model_path = model_path + 'autoencoder.pth'

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
        print(
            f"Autoencoder model training completed in {(datetime.now() - start_time).seconds} seconds, weights saved to {ae_model_path}")
    autoencoder_model = autoencoder_model.to(device)
    # Get reconstruction error on the test data
    reconstruction_error = ae.get_reconstruction_error(autoencoder_model, test_loader, device)
    # Set a threshold for anomaly detection
    threshold = np.percentile(reconstruction_error, 95)
    # Get predictions based on the threshold
    pred_ae = np.where(reconstruction_error > threshold, 1, 0)

    values_format = "d"  # The number format to use for the confusion matrix values
    print("\n--- Evaluation Results ---\n")
    plot_path = f'images/{model_name}/'
    # Isolation Forest evaluation
    print("Random Forest:")
    print(classification_report(y_test, pred_rf))
    cm_rf = ConfusionMatrixDisplay.from_predictions(y_test, pred_rf, display_labels=["Non-Fraud", "Fraud"],
                                                    values_format=values_format)
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(plot_path + 'RFcm.png')

    # XGBoost evaluation
    print("XGBoost:")
    print(classification_report(y_test, pred_xgb))
    cm_xgb = ConfusionMatrixDisplay.from_predictions(y_test, pred_xgb, display_labels=["Non-Fraud", "Fraud"],
                                                    values_format=values_format)
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(plot_path + 'XGBcm.png')

    # # K-NN evaluation
    # print("K-NN:")
    # print(classification_report(y_test, pred_knn))
    # cm_knn = ConfusionMatrixDisplay.from_predictions(y_test, pred_knn, display_labels=["Non-Fraud", "Fraud"],
    #                                                 values_format=values_format)
    # plt.title("K-NN Confusion Matrix")
    # plt.savefig(plot_path + 'KNNcm.png')

    # Transformer evaluation
    print("Transformer:")
    print(classification_report(y_test, preds_transformer))
    cm_transformer = ConfusionMatrixDisplay.from_predictions(y_test, preds_transformer, display_labels=["Non-Fraud", "Fraud"],
                                                             values_format=values_format)
    plt.title("Transformer Confusion Matrix")
    plt.savefig(plot_path + 'Transformercm.png')

    print("Isolation forest:")
    print(classification_report(y_test, pred_if))
    cm_isolation_forest = ConfusionMatrixDisplay.from_predictions(y_test, pred_if, display_labels=["Non-Fraud", "Fraud"],
                                                                  values_format=values_format)
    plt.title("Isolation Forest Confusion Matrix")
    plt.savefig(plot_path + 'IsolationForestcm.png')

    print("K-Means clustering:")
    print(classification_report(y_test, pred_kmeans))
    cm_kmeans = ConfusionMatrixDisplay.from_predictions(y_test, pred_kmeans, display_labels=["Non-Fraud", "Fraud"],
                                                        values_format=values_format)
    plt.title("K-Means clustering Confusion Matrix")
    plt.savefig(plot_path + 'Kmeanscm.png')

    print("Autoencoder:")
    print(classification_report(y_test, pred_ae))
    cm_autoencoder = ConfusionMatrixDisplay.from_predictions(y_test, pred_ae, display_labels=["Non-Fraud", "Fraud"],
                                                             values_format=values_format)
    plt.title("Autoencoder Confusion Matrix")
    plt.savefig(plot_path + 'Autoencodercm.png')

    plt.show()

    # Plot precision-recall and ROC for each model
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_precision_recall(y_test, pred_rf, 'Random Forest')
    plot_precision_recall(y_test, pred_xgb, 'XGBoost')
    # plot_precision_recall(y_test, pred_knn, 'K-NN')
    plot_precision_recall(y_test, preds_transformer, 'Transformer')
    plot_precision_recall(y_test, pred_if, 'Isolation Forest')
    plot_precision_recall(y_test, pred_kmeans, 'K-Means clustering')
    plot_precision_recall(y_test, pred_ae, 'Autoencoder')
    plt.legend()
    plt.subplot(1, 2, 2)
    plot_roc(y_test, pred_rf, 'Random Forest')
    plot_roc(y_test, pred_xgb, 'XGBoost')
    # plot_roc(y_test, pred_knn, 'K-NN')
    plot_roc(y_test, preds_transformer, 'Transformer')
    plot_roc(y_test, pred_if, 'Isolation Forest')
    plot_roc(y_test, pred_kmeans, 'K-Means clustering')
    plot_roc(y_test, pred_ae, 'Autoencoder')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path + 'PRAndRoc.png')




