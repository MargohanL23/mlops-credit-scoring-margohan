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

# ==============================================================
# ✅ FIXED: Konfigurasi MLflow Tracking ke DagsHub ✅
# ==============================================================

MLFLOW_TRACKING_URI = "https://dagshub.com/MargohanL23/mlops-credit-scoring-margohan.mlflow"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = "MargohanL23"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")  # di-set dari GitHub Actions Secrets

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"✅ MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"⚠ ERROR set URI: {e}")
    print("➡ MLflow offline mode")


# ✅ FIXED: Auto create experiment jika belum ada
EXPERIMENT_NAME = "Credit Scoring Tuning - MARGOHAN"
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ Experiment aktif: {EXPERIMENT_NAME}")
except:
    print("⚠ Experiment tidak ditemukan — membuat baru...")
    exp_id = mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


# Dataset path
PREPROCESSED_DATA_PATH = 'namadataset_preprocessing/clean_data.pkl'


def plot_confusion_matrix(cm, run_id):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    save_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(save_path)
    plt.close()
    return save_path


def train_and_log_model():
    print("\n🚀 Training model dimulai...")

    # Load Dataset
    try:
        data = joblib.load(PREPROCESSED_DATA_PATH)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        print("✅ Data loaded")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    except Exception as e:
        print(f"❌ ERROR load dataset: {e}")
        return

    # Hyperparameter Search
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

    # Logging ke MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"📌 MLflow Run ID: {run_id}")

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics(metrics)
        print("\n📊 Metrics Logged:")
        print(metrics)

        cm = confusion_matrix(y_test, y_pred)
        cm_file = plot_confusion_matrix(cm, run_id)
        mlflow.log_artifact(cm_file, "plots")
        os.remove(cm_file)

        # ✅ Log Feature Importance
        fi_file = f"feature_importance_{run_id}.csv"
        pd.Series(best_model.feature_importances_).to_csv(fi_file)
        mlflow.log_artifact(fi_file, "feature_importance")
        os.remove(fi_file)

        # ✅ Log Model
        model_file = "best_rf_model.pkl"
        joblib.dump(best_model, model_file)
        mlflow.log_artifact(model_file, "model")
        os.remove(model_file)

        print("\n✅ Model berhasil dicatat ke DagsHub ✅")


if __name__ == "__main__":
    train_and_log_model()
