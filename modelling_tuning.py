import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ===== KONFIGURASI MLFLOW ===== #
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

if TRACKING_URI:
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        print(f"✅ MLflow Tracking URI set to: {TRACKING_URI}")
    except Exception as e:
        print(f"⚠️ Gagal set tracking URI: {e}")
        print("➡️ MLflow akan berjalan dalam offline mode (local).")
else:
    print("⚠️ MLFLOW_TRACKING_URI tidak ditemukan. MLflow offline mode.")

mlflow.set_experiment("Credit Scoring Tuning - MARGOHAN")

# Path dataset fix ✅ tanpa double slash
PREPROCESSED_DATA_PATH = 'namadataset_preprocessing/clean_data.pkl'


def plot_confusion_matrix(cm, run_id):
    """Membuat dan menyimpan Confusion Matrix sebagai artefak."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    save_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(save_path)
    plt.close()
    return save_path


def train_and_log_model():
    """Training model + Log hasil ke MLflow"""

    print("\n🚀 Mulai proses training model...")

    # ===== 1. Load Data ===== #
    try:
        data = joblib.load(PREPROCESSED_DATA_PATH)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        print("✅ Data preload berhasil!")

        # Encoding label target
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    except Exception as e:
        print(f"❌ ERROR: Tidak bisa load file data: {PREPROCESSED_DATA_PATH}")
        print(f"Detail: {e}")
        return

    # ===== 2. Config Randomized Search Hyperparameters ===== #
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    rf = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    # ===== 3. MLflow Logging ===== #
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"📌 MLflow Run ID: {run_id}")

        # Training
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Prediction
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # ===== Log Metrics ===== #
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics(metrics)

        print("\n📊 Metrik Berhasil Dilog:")
        print(metrics)

        # ===== Log Confusion Matrix ===== #
        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(cm, run_id)
        mlflow.log_artifact(cm_path, "evaluation_plots")
        os.remove(cm_path)
        print("✅ Confusion Matrix terlog ke MLflow")

        # ===== Log Feature Importance ===== #
        fi_path = f"feature_importance_{run_id}.csv"
        feature_importance = pd.Series(best_model.feature_importances_)
        feature_importance.to_csv(fi_path)
        mlflow.log_artifact(fi_path, "model_metadata")
        os.remove(fi_path)
        print("✅ Feature importance terlog")

        # ===== Log Model ===== #
        model_path = "best_random_forest_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, "model_artifact")
        os.remove(model_path)

        print("\n✅ Model terbaik sukses dilog ke DagsHub/MLflow ✅")


if __name__ == "__main__":
    train_and_log_model()
