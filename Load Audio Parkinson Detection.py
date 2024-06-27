import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import StandardScaler
import joblib

# Fungsi untuk ekstraksi formants menggunakan Praat melalui Parselmouth
def extract_formants(audio_path, n_formants=8):
    snd = parselmouth.Sound(audio_path)
    formant = call(snd, "To Formant (burg)", 0.0, 8.0, 5500, 0.025, 50.0)
    
    formant_features = []
    for i in range(1, n_formants + 1):
        f_values = []
        for t in range(1, formant.get_number_of_frames() + 1):
            f = call(formant, "Get value at time", i, formant.get_time_from_frame_number(t), "Hertz", "Linear")
            if not np.isnan(f):
                f_values.append(f)
        if f_values:
            formant_features.append(np.mean(f_values))
        else:
            formant_features.append(0)
    
    return np.array(formant_features)

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

# Fungsi untuk memuat model dan melakukan prediksi pada data baru
def predict_new_audio(audio_path, model_path, scaler_path):
    # Load model
    model = load_model(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Ekstraksi fitur dari file audio baru
    features = extract_formants(audio_path)
    
    # Normalisasi data
    features = features.reshape(1, -1)
    features, _ = normalize_data(features, scaler)
    
    # Prediksi
    prediction = model.predict(features)
    
    # Konversi prediksi ke label
    predicted_label = np.argmax(prediction, axis=1)
    
    return predicted_label, prediction

if __name__ == '__main__':
    # Path ke model dan scaler yang telah disimpan
    model_path = "C:/Users/farha/OneDrive/Documents/SKRIPSI/Kodingan Skripsi/Formant/best_model_terbaru_v2_LENGKAP_f8_nospatial.h5"
    scaler_path = 'scaler_formant_gpt4_LENGKAP_f8_nospatial.pkl'
    
    # Path ke file audio baru yang akan diprediksi
    new_audio_paths = [
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/06dataset_adam74.wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/06dataset_adam75.wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/06dataset_adam76.wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/06dataset_adam77.wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/06dataset_adam78.wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/LSVT LOUD Speech Therapy for Parkinson disease (1).wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/LSVT LOUD Speech Therapy for Parkinson disease 3 (1).wav",
        "C:/Users/farha/OneDrive/Documents/SKRIPSI/Data Uji/Parkinson/parkinson (1).wav"
    ]
    
    for audio_path in new_audio_paths:
        predicted_label, prediction = predict_new_audio(audio_path, model_path, scaler_path)
        label_map = {0: "Non-Parkinson", 1: "Parkinson"}
        result = label_map[predicted_label[0]]
        print(f"Predicted Label for {os.path.basename(audio_path)}: {result}")
        print(f"Prediction Confidence: {prediction}")
