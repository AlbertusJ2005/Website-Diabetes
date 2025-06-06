from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model dan scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Ambil fitur dari input
    features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['bloodpressure']),
        float(data['skinthickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetespedigreefunction']),
        float(data['age']),
    ]

    # Preprocessing
    features_array = np.array([features])
    scaled_features = scaler.transform(features_array)

    # Prediksi probabilitas
    probability = model.predict_proba(scaled_features)[0][1]  # probabilitas kelas "1" (positif diabetes)
    prediction = model.predict(scaled_features)[0]
    result = "Diabetes" if prediction == 1 else "Tidak Diabetes"

    return jsonify({
        'prediction': result,
        'probability': round(probability * 100, 2)  # persen, dibulatkan 2 angka
    })

if __name__ == '__main__':
    app.run(debug=True)
