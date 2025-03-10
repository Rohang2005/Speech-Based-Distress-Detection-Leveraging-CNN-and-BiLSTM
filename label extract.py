import os
import pandas as pd

DATASET_PATH = "CREMA-D/"

EMOTIONS = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
DISTRESS_LABELS = {"angry", "disgust", "fear"}

data = []
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        parts = file.split("_")
        emotion_code = parts[2]
        if emotion_code in EMOTIONS:
            label = 1 if EMOTIONS[emotion_code] in DISTRESS_LABELS else 0
            data.append((file, EMOTIONS[emotion_code], label))

df = pd.DataFrame(data, columns=["filename", "emotion", "distress?"])
df.to_csv("labels.csv", index=False)
