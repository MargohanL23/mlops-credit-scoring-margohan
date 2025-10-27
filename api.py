import os
import sys
import json
import joblib
from flask import Flask, request, jsonify
import mlflow
import numpy as np

# --- Konfigurasi MLflow (Diambil dari Environment Variables) ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# Nama experiment harus sama
EXPERIMENT_NAME = "Credit Scoring Tuning - MARGOHAN"
# Nama folder artifact dan nama file model saat kamu log_artifact di modelling_tuning.py
MODEL_ARTIFACT_PATH = "model" 
MODEL_FILE_NAME = "best_rf_model.pkl" 

app = Flask(__name__)
model = None # Model diinisialisasi sebagai None global

# --- Fungsi Pemuatan Model ---
def load_model_from_mlflow():
    """Mengunduh model terbaik (Run terakhir) dari MLflow/DagsHub."""
    
    # Ketika dijalankan oleh Gunicorn/Production, variabel ini harus ada
    if not MLFLOW_TRACKING_URI:
        print("❌ ERROR: MLFLOW_TRACKING_URI tidak disetel. Gagal memuat model.")
        # Mengembalikan None agar server bisa tetap hidup (meskipun error)
        return None 

    print(f"✅ MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # 1. Cari Experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
             print(f"❌ ERROR: Experiment '{EXPERIMENT_NAME}' tidak ditemukan. Pastikan sudah di-run sebelumnya.")
             return None
             
        # 2. Cari Run Terbaik/Terakhir
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            print("❌ ERROR: Tidak ada run yang ditemukan dalam experiment.")
            return None
            
        latest_run_id = runs.iloc[0]['run_id']
        print(f"✅ Run ID Terbaik ditemukan: {latest_run_id}")
        
        # 3. Unduh Artifact Model dari DagsHub
        local_path = mlflow.artifacts.download_artifacts(
            run_id=latest_run_id,
            artifact_path=f"{MODEL_ARTIFACT_PATH}/{MODEL_FILE_NAME}" 
        )
        
        # 4. Muat Model
        # Catatan: local_path adalah path file yang sudah diunduh ke container
        loaded_model = joblib.load(local_path)
        print(f"✅ Model terbaik berhasil diunduh dan dimuat dari DagsHub/MLflow.")
        return loaded_model

    except Exception as e:
        print(f"❌ FATAL ERROR: Gagal memuat model dari MLflow/DagsHub. Detail: {e}")
        return None

# Kode inisialisasi dipindahkan ke blok __main__


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk mendapatkan prediksi skor kredit."""
    if model is None:
        # Jika model gagal dimuat saat startup, berikan error 500
        return jsonify({"error": "Model belum dimuat. Server gagal inisialisasi."}), 500

    try:
        # Data yang diharapkan adalah array/list of features (numerik)
        data = request.get_json(force=True)
        
        if not isinstance(data, list) or not data:
            return jsonify({"error": "Payload harus berupa list berisi fitur-fitur numerik."}), 400

        features = np.array(data)
        
        # Lakukan reshape jika hanya satu instance 
        if features.ndim == 1:
            features = features.reshape(1, -1) 

        # Prediksi
        prediction = model.predict(features)
        proba = model.predict_proba(features)

        result = [
            {
                "prediction_label": int(pred),
                "probability_default": float(prob[1]), # Probabilitas kelas 1 (Default)
                "status": "DEFAULT" if pred == 1 else "NON_DEFAULT"
            }
            for pred, prob in zip(prediction, proba)
        ]

        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses data: {e}. Pastikan format data benar (numerik dan jumlah fitur sesuai)."},
                       {"received_data_structure": str(type(request.get_json()))}), 400

# Endpoint status (Health Check)
@app.route('/', methods=['GET'])
def health_check():
    """Endpoint untuk memeriksa status server."""
    status = "READY" if model is not None else "MODEL_LOADING_FAILED"
    return jsonify({
        "status": status, 
        "model": "RandomForestClassifier",
        "message": "Credit Scoring API operational."
    })


if __name__ == '__main__':
    # Global model diisi sebelum app.run() dipanggil.
    model = load_model_from_mlflow()
    if model:
        print("Server API siap melayani permintaan!")

    # Server dijalankan pada port 5000, host 0.0.0.0 agar bisa diakses dari luar container
    print("Mencoba menjalankan server Flask...")
    # Catatan: Ketika Gunicorn dijalankan (di Dockerfile), blok ini tidak dieksekusi,
    # tetapi kode Gunicorn secara otomatis akan memuat model karena model diinisialisasi secara global (Top-Level Code).
    app.run(host='0.0.0.0', port=5000)