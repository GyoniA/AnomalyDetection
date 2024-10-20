from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
import data_loader
from anomaly import model_path
app = Flask(__name__)

X_train_scaled, X_test_scaled, Y_train, Y_test = data_loader.load_cic_unsw_data(binary=True)

MODEL_PATHS = {
    'Isolation Forest': f'{model_path}/iso_forest.pkl',
    'Local Outlier Factor': f'{model_path}/lof.pkl',
    'One-Class SVM': f'{model_path}/ocsvm.pkl',
    'K-Means': f'{model_path}/kmeans.pkl',
    # TODO: Add Autoencoder and Transformer paths
}

PLOT_DIR = 'static/plots/'
os.makedirs(PLOT_DIR, exist_ok=True)


def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(path):
        return joblib.load(path)
    return None


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

    predictions = model.predict(X_test_scaled)
    predictions = np.where(predictions == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    cm_path, pr_path, roc_path = generate_plots(Y_test, predictions, model_name)
    classification_rep = classification_report(Y_test, predictions)

    return render_template('results.html', model_name=model_name, classification_rep=classification_rep,
                           cm_path=cm_path, pr_path=pr_path, roc_path=roc_path)


if __name__ == '__main__':
    app.run(debug=True)