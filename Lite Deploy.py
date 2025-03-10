import numpy as np
import librosa
import joblib
import tensorflow as tf
import speech_recognition as sr
import tempfile
import os

interpreter = tf.lite.Interpreter(model_path="speech_distress_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load("scaler.save")

with open("feature_columns.txt", "r") as f:
    feature_columns = f.read().splitlines()

keywords = ["help", "emergency", "save me", "please help", "danger", "call police","stop"]
DISTRESS_THRESHOLD = 0.6

def extract_features(y, sr):
    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    features.extend(mfcc_mean)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features.append(zcr)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features.append(spec_centroid)
    rms = np.mean(librosa.feature.rms(y=y))
    features.append(rms)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    features.extend(chroma_mean)
    return np.array(features)

def transcribe_audio(file_path):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).lower()
    except Exception:
        return ""

def predict_distress(file_path):
    try:
        y, sr_audio = librosa.load(file_path, sr=None)
        features = extract_features(y, sr_audio)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None, None

    if len(features) < len(feature_columns):
        features = np.pad(features, (0, len(feature_columns) - len(features)))
    else:
        features = features[:len(feature_columns)]

    try:
        scaled_features = scaler.transform([features])
        reshaped_input = scaled_features.reshape((1, len(feature_columns), 1)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], reshaped_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        return prediction, features
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None

def detect_distress_once(duration=5):
    recognizer = sr.Recognizer()
    print(f"\nRecording {duration} seconds of audio for distress detection...\n")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, phrase_time_limit=duration)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio.get_wav_data())
        temp_path = f.name

    transcript = transcribe_audio(temp_path)
    prediction, _ = predict_distress(temp_path)
    keyword_hit = any(word in transcript for word in keywords)

    print(f"\nTranscribed Text: {transcript if transcript else '[Unable to transcribe]'}")
    if prediction is not None:
        if prediction > DISTRESS_THRESHOLD and keyword_hit:
            print("HIGH ALERT: Distress detected via voice + keywords.")
        elif prediction > DISTRESS_THRESHOLD:
            print("Voice distress detected.")
        elif keyword_hit:
            print("Keyword-based distress detected.")
        else:
            print("No distress detected.")
    os.remove(temp_path)

if __name__ == "__main__":
    detect_distress_once()
