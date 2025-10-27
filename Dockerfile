# Gunakan base image Python 3.12 slim
FROM python:3.12-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Instal semua dependencies. Gunakan --no-cache-dir untuk instalasi yang lebih bersih.
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh folder dan script proyek ke dalam container
# Kita asumsikan struktur folder seperti ini:
# /membangun_model
#   |-- modelling_tuning.py
#   |-- namadataset_preprocessing/ (Berisi clean_data.pkl)
#   |-- requirements.txt
#   |-- Dockerfile
COPY . .

# Perintah utama saat container dijalankan: jalankan script pelatihan
# Perlu diperhatikan: Saat CI/CD, kita perlu memastikan 
# DAGSHUB_TOKEN, MLFLOW_TRACKING_URI, dll sudah disetel.
CMD ["python", "modelling_tuning.py"]