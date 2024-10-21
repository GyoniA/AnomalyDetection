import torch
from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
import data_loader
from anomaly import model_path, device
import auto_encoder as ae
from transformer import TransformerClassifier, get_optimal_threshold

app = Flask(__name__)

X_train_scaled, X_test_scaled, Y_train, Y_test = data_loader.load_cic_unsw_data(binary=True)
train_loader, test_loader = data_loader.create_dataloader(X_train_scaled, X_test_scaled, use_gpu=torch.cuda.is_available())
class_train_loader, class_test_loader = data_loader.create_classification_dataloader(X_train_scaled, X_test_scaled, Y_train, Y_test)

MODEL_PATHS = {
    'Isolation Forest': f'{model_path}/iso_forest.pkl',
    'Local Outlier Factor': f'{model_path}/lof.pkl',
    'One-Class SVM': f'{model_path}/ocsvm.pkl',
    'K-Means': f'{model_path}/kmeans.pkl',
    'Autoencoder': f'{model_path}/autoencoder.pth',
    'Transformer': f'{model_path}/transformer.pth',
}

PLOT_DIR = 'static/plots/'
os.makedirs(PLOT_DIR, exist_ok=True)


def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if model_name == 'Autoencoder':
        model = ae.Autoencoder(input_dim=X_train_scaled.shape[1])
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        return model
    if model_name == 'Transformer':
        model = TransformerClassifier(input_dim=X_train_scaled.shape[1])
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        return model
    if path and os.path.exists(path):
        return joblib.load(path)
    return None

def get_predictions(model, model_name):
    if model_name == 'Autoencoder':
        reconstruction_error = ae.get_reconstruction_error(model, test_loader, device)
        # Set a threshold for anomaly detection
        threshold = np.percentile(reconstruction_error, 95)
        # Get predictions based on the threshold
        pred_ae = np.where(reconstruction_error > threshold, 1, 0)
        return pred_ae
    if model_name == 'Transformer':
        model.eval()
        all_labels_transformer = []
        all_probs_transformer = []
        with torch.no_grad():
            for inputs, labels in class_test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)

                all_labels_transformer.extend(labels.cpu().numpy())
                all_probs_transformer.extend(probs.cpu().numpy())
        # Determine optimal threshold
        optimal_threshold, best_f1 = get_optimal_threshold(np.array(all_labels_transformer), np.array(all_probs_transformer))
        preds_transformer = (np.array(all_probs_transformer) >= optimal_threshold).astype(int)
        return preds_transformer
    predictions = model.predict(X_test_scaled)
    predictions = np.where(predictions == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
    return predictions


def generate_plots(y_test, predictions, model_name):
    """
    Generate and save confusion matrix, PR, and ROC plots, for the given model.
    """
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Non-Fraud", "Fraud"], values_format='d')
    plt.title(f'{model_name} Confusion Matrix')
    cm_path = os.path.join(PLOT_DIR, f'{model_name}_cm.png')
    plt.savefig(cm_path)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, marker='.', label=f'{model_name} AUPRC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    pr_path = os.path.join(PLOT_DIR, f'{model_name}_pr.png')
    plt.savefig(pr_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    roc_path = os.path.join(PLOT_DIR, f'{model_name}_roc.png')
    plt.savefig(roc_path)
    plt.close()

    return cm_path, pr_path, roc_path


@app.route('/')
def index():
    return render_template('index.html', models=list(MODEL_PATHS.keys()))


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate the selected model on the test dataset and display the results.
    """
    model_name = request.form.get('model')
    model = load_model(model_name)

    if model is None:
        return "Model not found!", 404

    predictions = get_predictions(model, model_name)

    cm_path, pr_path, roc_path = generate_plots(Y_test, predictions, model_name)
    classification_rep = classification_report(Y_test, predictions)

    return render_template('results.html', model_name=model_name, classification_rep=classification_rep,
                           cm_path=cm_path, pr_path=pr_path, roc_path=roc_path)


if __name__ == '__main__':
    app.run(debug=True)