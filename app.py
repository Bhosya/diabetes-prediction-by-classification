from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
try:
    with open("model.pkl", "rb") as f:
        models = pickle.load(f)
        model_dt = models[0]   # Decision Tree
        model_svc = models[1]  # SVM

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("✅ Berhasil memuat model.pkl dan scaler.pkl!")
except FileNotFoundError:
    print("❌ Error: File model.pkl atau scaler.pkl tidak ditemukan.")

# Route utama untuk menampilkan halaman web index.html
@app.route("/")
def home():
    return render_template("index.html")

# Route untuk memproses prediksi (menerima data dari form HTML)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Mengambil data dari form HTML
        fitur_input = np.array([[
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]])
        
        # Standardisasi data
        fitur_scaled = scaler.transform(fitur_input)
        
        # Prediksi dengan kedua model
        prediksi_dt = model_dt.predict(fitur_scaled)[0]
        prediksi_svc = model_svc.predict(fitur_scaled)[0]
        
        hasil_dt = "Positif Diabetes" if prediksi_dt == 1 else "Negatif Diabetes"
        hasil_svc = "Positif Diabetes" if prediksi_svc == 1 else "Negatif Diabetes"
        
        # Kirim hasil kembali ke halaman web
        return render_template("index.html", 
                               hasil_dt=hasil_dt, 
                               hasil_svc=hasil_svc,
                               input_data=request.form)

    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)