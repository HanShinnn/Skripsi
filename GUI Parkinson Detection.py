import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import numpy as np
from tensorflow.keras.models import load_model
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import StandardScaler
import joblib
import time
from threading import Thread

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

class ParkinsonDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Sistem Deteksi Parkinson")
        master.geometry("650x420")
        master.configure(bg="#ff69b4")

        self.page1 = Page1(master, self)
        self.page2 = Page2(master, self)

        self.show_page1()

    def show_page1(self):
        self.page2.hide()
        self.page1.show()

    def show_page2(self, prediction_result, confidence):
        self.page1.hide()
        self.page2.show(prediction_result, confidence)

class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()
        self.frame.configure(bg="#ff69b4")

        self.label = tk.Label(self.frame, text="Rekam suara anda", font=("Helvetica", 28, "bold"), bg="#ff69b4", fg="white")
        self.record_button = tk.Button(self.frame, text="Mulai Rekam", font=("Helvetica", 28, "bold"), bg="black", fg="white", command=self.record_audio)

        self.label.pack(pady=20)
        self.record_button.pack(pady=80)

    def show(self):
        self.frame.pack()

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):
        for i in range(3, 0, -1):
            self.label.config(text=f"Perekaman dimulai dalam {i}")
            self.master.update()
            time.sleep(1)

        self.label.config(text="Perekaman suara...")
        self.master.update()

        recording_thread = Thread(target=self.simulate_recording)
        recording_thread.start()

    def simulate_recording(self):
        try:
            duration = 4
            file_name = "recorded_audio.wav"

            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=44100,
                            input=True,
                            frames_per_buffer=1024)

            frames = []

            for i in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))

            self.label.config(text="Rekaman suara anda berhasil")
            self.master.update()

            model_path = "C:/Users/farha/OneDrive/Documents/SKRIPSI/Kodingan Skripsi/Formant/best_model_terbaru_v2_LENGKAP_f8_nospatial.h5"
            scaler_path = 'scaler_formant_gpt4_LENGKAP_f8_nospatial.pkl'

            predicted_label, prediction = predict_new_audio(file_name, model_path, scaler_path)
            label_map = {0: "Non-Parkinson", 1: "Parkinson"}
            result = label_map[predicted_label[0]]
            confidence = prediction[0][predicted_label[0]] * 100

            self.app.show_page2(result, confidence)

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat merekam audio: {str(e)}")

class Page2:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()
        self.frame.configure(bg="#ff69b4")

        self.label = tk.Label(self.frame, text="Hasil Prediksi:", font=("Helvetica", 28, "bold"), bg="#ff69b4", fg="white")
        self.prediction_label = tk.Label(self.frame, text="", font=("Helvetica", 24, "bold"), bg="#ff69b4", fg="white")
        self.confidence_label = tk.Label(self.frame, text="", font=("Helvetica", 20, "bold"), bg="#ff69b4", fg="white")

        self.back_button = tk.Button(self.frame, text="Back", font=("Helvetica", 25, "bold"), bg="black", fg="white", command=app.show_page1)

        self.label.pack(pady=20)
        self.prediction_label.pack(pady=20)
        self.confidence_label.pack(pady=20)
        self.back_button.pack(pady=40)

    def show(self, prediction_result, confidence):
        self.prediction_label.config(text=f"{prediction_result}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        self.frame.pack()
        self.frame.configure(bg="#ff69b4")

    def hide(self):
        self.frame.pack_forget()

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

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkinsonDetectionApp(root)
    root.mainloop()
