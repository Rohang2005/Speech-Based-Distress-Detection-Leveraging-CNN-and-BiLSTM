from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
import librosa
import soundfile as sf 
import joblib
import tensorflow as tf
import speech_recognition as sr
import tempfile
import os

app = FastAPI(title="Distress Detection API")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

interpreter = tf.lite.Interpreter(model_path="speech_distress_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load("scaler.save")
with open("feature_columns.txt", "r") as f:
    feature_columns = f.read().splitlines()

keywords = ["help", "emergency", "save me", "please help", "danger", "call police", "stop"]
DISTRESS_THRESHOLD = 0.6

def convert_audio_to_wav(input_path):
    """
    Convert various audio formats (MP3, M4A, OGG, FLAC) to WAV using librosa & soundfile.
    """
    y, sr = librosa.load(input_path, sr=None)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_wav.name, y, sr)
    return temp_wav.name

def extract_features(y, sr):
    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    features.extend(mfcc_mean)
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma.T, axis=0))
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
    except Exception:
        return None
    
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
        return prediction
    except Exception:
        return None

@app.post("/detect_distress")
def detect_distress(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_audio:
        temp_audio.write(file.file.read())
        temp_path = temp_audio.name

    if not file.filename.lower().endswith(".wav"):
        converted_path = convert_audio_to_wav(temp_path)
        os.remove(temp_path)
        temp_path = converted_path

    transcript = transcribe_audio(temp_path)
    distress_prediction = predict_distress(temp_path)
    keyword_hit = any(word in transcript for word in keywords)
    os.remove(temp_path)

    if distress_prediction is not None and distress_prediction > DISTRESS_THRESHOLD and keyword_hit:
        distress_type = "Both voice-based and keyword-based distress detected."
    elif distress_prediction is not None and distress_prediction > DISTRESS_THRESHOLD:
        distress_type = "Voice-based distress detected."
    elif keyword_hit:
        distress_type = "Keyword-based distress detected."
    else:
        distress_type = "None"

    return {
        "transcript": transcript if transcript else "[Unable to transcribe]",
        "distress_type": distress_type
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Welcome to Distress Detection API</title>
    </head>
    <body>
        <h1>Welcome to the Distress Detection API!</h1>
        <p>To test the API, go to <a href="/upload">/upload</a> to upload an audio file.</p>
    </body>
    </html>
    """

@app.get("/upload", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
    <head>
        <title>Upload Audio for Distress Detection</title>
    </head>
    <body>
        <h1>Upload an Audio File (MP3, WAV, M4A, OGG, FLAC)</h1>
        <form action="/detect_distress" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept=".mp3,.wav,.m4a,.ogg,.flac">
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """
