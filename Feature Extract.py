import os
import librosa
import numpy as np
import pandas as pd
import tqdm

DATASET_PATH = "CREMA-D/"
LABELS_CSV = "labels.csv"

df = pd.read_csv(LABELS_CSV)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    min_signal_length = 512
    if len(y) < min_signal_length:
        y = np.pad(y, (0, min_signal_length - len(y)), mode='constant')

    n_fft = min(512, len(y))
    hop_length = n_fft // 2

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).T, axis=0)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    pitch = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0

    harmonic, percussive = librosa.effects.hpss(y)
    hnr = np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-6)

    frame_length_env = 2048
    hop_length_env = 512
    energy_env = np.array([
        sum(abs(y[i:i+frame_length_env]**2))
        for i in range(0, len(y), hop_length_env)
    ])
    energy_spike = np.max(energy_env)

    return np.hstack([mfcc, chroma, mel, contrast, tonnetz, zcr, rms, pitch, hnr, energy_spike])

features = []
for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    file_path = os.path.join(DATASET_PATH, row["filename"])
    try:
        feature_vector = extract_features(file_path)
        features.append([row["filename"]] + feature_vector.tolist() + [row["distress?"]])
    except Exception as e:
        print(f"Error processing {row['filename']}: {e}")

mfcc_cols = [f"mfcc_{i}" for i in range(13)]
chroma_cols = [f"chroma_{i}" for i in range(12)]
mel_cols = [f"mel_{i}" for i in range(128)]
contrast_cols = [f"contrast_{i}" for i in range(7)]
tonnetz_cols = [f"tonnetz_{i}" for i in range(6)]
zcr_col = ["zcr"]
rms_col = ["rms"]
pitch_col = ["pitch"]
hnr_col = ["hnr"]
energy_spike_col = ["energy_spike"]

columns = ["filename"] + mfcc_cols + chroma_cols + mel_cols + contrast_cols + tonnetz_cols + zcr_col + rms_col + pitch_col + hnr_col + energy_spike_col + ["distress?"]
features_df = pd.DataFrame(features, columns=columns)

features_df.to_csv("features.csv", index=False)
