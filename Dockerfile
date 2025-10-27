# Gunakan Python 3.12 (disarankan)
FROM python:3.12-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Instal semua dependencies, termasuk Flask dan Gunicorn yang baru ditambahkan
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh file kode (termasuk api.py) dan data ke dalam container
COPY . .

# Expose port yang digunakan Gunicorn
EXPOSE 5000

# Perintah utama: Jalankan server Gunicorn untuk Flask API
# Ini akan menjalankan server pada port 5000, yang siap menerima request prediksi.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]