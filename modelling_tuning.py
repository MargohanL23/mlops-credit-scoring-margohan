import os
import joblib
import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder 

# --- KONFIGURASI DAGSHUB/MLFLOW ---

REPO_OWNER = "MargohanL23" 
REPO_NAME = "mlops-credit-scoring-margohan" 

# Inisialisasi DagsHub dan Atur Tracking URI
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
# Menggunakan mlflow.get_tracking_uri() untuk kompatibilitas versi
mlflow.set_tracking_uri(mlflow.get_tracking_uri()) 
mlflow.set_experiment("Credit Scoring Tuning - MARGOHAN")

# Path ke artefak Kriteria 1
PREPROCESSED_DATA_PATH = 'namadataset_preprocessing/clean_data.pkl'

def plot_confusion_matrix(cm, run_id):
    """Membuat plot Confusion Matrix dan menyimpannya sebagai artefak."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Simpan plot ke file lokal
    plot_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_and_log_model():
    """Melatih model, melakukan tuning, dan mencatat secara manual ke MLflow."""
    try:
        # 1. Load Data Bersih
        data = joblib.load(PREPROCESSED_DATA_PATH)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        # --- PERBAIKAN: ENCODING LABEL TARGET ---
        # Mengubah label string ('bad', 'good') menjadi numerik (0, 1)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        print("Data bersih berhasil dimuat dan label di-encode.")
    except Exception as e:
        print(f"ERROR: Gagal memuat data bersih. Pastikan path '{PREPROCESSED_DATA_PATH}' benar.")
        print(f"Detail error: {e}")
        return

    # 2. Definisikan Hyperparameter Tuning
    # Parameter untuk tuning (Contoh menggunakan RandomForest)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Inisialisasi Model Dasar
    rf = RandomForestClassifier(random_state=42)
    
    # Randomized Search untuk Tuning
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=5, 
        scoring='f1', # Akan mencari label 1 sebagai positive
        random_state=42, 
        n_jobs=-1
    )
    
    # --- 3. MULAI MLFLOW MANUAL LOGGING ---
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        
        # Latih model dengan tuning
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Prediksi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] # Probabilitas untuk ROC AUC

        # --- 4. LOG PARAMETER ---
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("data_split", "80/20")
        
        # --- 5. HITUNG DAN LOG METRIK ---
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            # Metrik Tambahan 1: ROC AUC Score 
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        mlflow.log_metrics(metrics)
        print("\nMetrik Dasar Berhasil Dilog:")
        print(metrics)

        # --- 6. LOG ARTEFAK TAMBAHAN ---
        # Artefak 1: Confusion Matrix Plot 
        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(cm, run_id)
        mlflow.log_artifact(cm_path, "evaluation_plots")
        os.remove(cm_path) # Hapus file lokal setelah diunggah
        print(f"Artefak 1 (Confusion Matrix Plot) berhasil dilog.")

        # Artefak 2: Feature Importance 
        feature_importance = pd.Series(best_model.feature_importances_)
        feature_importance_path = f"feature_importance_{run_id}.csv"
        feature_importance.to_csv(feature_importance_path)
        mlflow.log_artifact(feature_importance_path, "model_metadata")
        os.remove(feature_importance_path)
        print(f"Artefak 2 (Feature Importance) berhasil dilog.")

        # --- 7. LOG MODEL ---        
        # 1. Simpan model terbaik secara lokal menggunakan joblib
        model_path = "best_random_forest_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"Model terbaik disimpan secara lokal di {model_path}.")

        # 2. Log file model sebagai artefak ke MLflow/DagsHub
        mlflow.log_artifact(model_path, "model_artifact")
        os.remove(model_path) # Hapus file lokal setelah diunggah
        
        print("\nModel terbaik berhasil dilog ke MLflow/DagsHub sebagai artefak.")

if __name__ == "__main__":
    train_and_log_model()