import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
from keras.utils import to_categorical
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Dense, MaxPooling1D, GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ProcessPoolExecutor
import itertools

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

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.show()

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        augmented_X.append(x)
        augmented_Y.append(y)
        noise = np.random.normal(0, 0.005, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)
        
        pitch_shift = np.random.randint(-3, 3)
        augmented_X.append(librosa.effects.pitch_shift(x.flatten(), sr=22050, n_steps=pitch_shift).reshape(x.shape))
        augmented_Y.append(y)
        
    return np.array(augmented_X), np.array(augmented_Y)

def build_enhanced_model(input_shape):
    model = Sequential([
        Conv1D(128, padding='same', kernel_size=3, input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1),
        Dropout(0.4),
        Conv1D(256, padding='same', kernel_size=3),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1),
        Dropout(0.4),
        Conv1D(512, padding='same', kernel_size=3),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1),
        Dropout(0.4),
        GlobalAveragePooling1D(),
        Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.summary()
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(healthy_folder, parkinson_folder):
    healthy_paths = [os.path.join(healthy_folder, f) for f in os.listdir(healthy_folder)]
    parkinson_paths = [os.path.join(parkinson_folder, f) for f in os.listdir(parkinson_folder)]
    audio_paths = healthy_paths + parkinson_paths
    labels = [0] * len(healthy_paths) + [1] * len(parkinson_paths)
    
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(extract_formants, audio_paths))
    
    features = np.array(features)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features, np.array(labels), healthy_paths, parkinson_paths

def process_and_train(X, Y, epochs=350):
    le = LabelEncoder()
    dataset_y_encoded = le.fit_transform(Y)
    dataset_y_onehot = to_categorical(dataset_y_encoded)
    X, scaler = normalize_data(X)
    
    # Simpan scaler
    joblib.dump(scaler, 'scaler_formant_gpt4_LENGKAP_f8_nospatial.pkl')
    
    X_augmented, Y_augmented = augment_data(X, dataset_y_onehot)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_augmented, Y_augmented, test_size=0.2, random_state=42, stratify=Y_augmented)
    
    model = build_enhanced_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint("best_model_terbaru_v2_LENGKAP_f8_nospatial.h5", monitor='val_accuracy', save_best_only=True)
    
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    return model, history

def generate_confusion_matrix(model, X, y):
    # Normalize data (X_scaled only, not returning scaler)
    X_scaled, scaler = normalize_data(X)
    # Predict classes
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    # Decode labels from one-hot encoding
    y_true = np.argmax(y, axis=1)
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.Blues  # Add this line to define the colormap
    classes = np.unique(y_true)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def generate_classification_report(model, X, y):
    # Normalize data (X_scaled only, not returning scaler)
    X_scaled, scaler = normalize_data(X)
    # Predict classes
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    # Decode labels from one-hot encoding
    y_true = np.argmax(y, axis=1)
    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)
    return report

def generate_f1_score(model, X, y):
    # Normalize data (X_scaled only, not returning scaler)
    X_scaled, scaler = normalize_data(X)
    # Predict classes
    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    # Decode labels from one-hot encoding
    y_true = np.argmax(y, axis=1)
    # Calculate F1-score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-score: {f1}")
    return f1

if __name__ == '__main__':
    healthy_folder = "C:/Users/farha/OneDrive/Documents/SKRIPSI/Kodingan Skripsi/Formant/Dataset FIX LENGKAP/Sehat"
    parkinson_folder = "C:/Users/farha/OneDrive/Documents/SKRIPSI/Kodingan Skripsi/Formant/Dataset FIX LENGKAP/Parkinson"
    X, Y, healthy_paths, parkinson_paths = load_data(healthy_folder, parkinson_folder)
    trained_model, history = process_and_train(X, Y)

    Y_onehot = to_categorical(Y)
    print(X)
    X, _ = normalize_data(X)
    print(X)

    plot_history(history)
    evaluation = trained_model.evaluate(X, Y_onehot)
    print(f"Final Loss: {evaluation[0]}")
    print(f"Final Accuracy: {evaluation[1]}")

    best_model = tf.keras.models.load_model("best_model_terbaru_v2_LENGKAP_f8_nospatial.h5")
    best_evaluation = best_model.evaluate(X, Y_onehot)
    print(f"Best Model Final Loss: {best_evaluation[0]}")
    print(f"Best Model Final Accuracy: {best_evaluation[1]}")

    # Generate and plot confusion matrix
    generate_confusion_matrix(trained_model, X, Y_onehot)

    # Generate classification report
    classification_report = generate_classification_report(trained_model, X, Y_onehot)

    # Generate F1-score
    f1_score = generate_f1_score(trained_model, X, Y_onehot)