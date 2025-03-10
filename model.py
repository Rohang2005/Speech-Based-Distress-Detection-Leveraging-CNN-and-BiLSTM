import numpy as np
import pandas as pd
import joblib
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Dropout, BatchNormalization,
                                     MaxPooling1D, SpatialDropout1D, Bidirectional,
                                     GlobalAveragePooling1D, GlobalMaxPooling1D,
                                     Input, Multiply, Reshape, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * K.pow(1. - pt, gamma) * K.log(pt)
        return K.mean(loss)
    return focal_loss_fixed

def attention_layer(inputs):
    attention = Dense(inputs.shape[-1], activation='softmax')(inputs)
    return Multiply()([inputs, attention])

df = pd.read_csv("features.csv")

X = df.drop(columns=["filename", "distress?"], errors='ignore')
y = df["distress?"].values.astype(np.int32)

feature_columns = X.columns.tolist()
with open("feature_columns.txt", "w") as f:
    f.write("\n".join(feature_columns))

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))
with open("class_weights.json", "w") as f:
    json.dump(class_weights_dict, f)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
joblib.dump(scaler, "scaler.save")

X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)

def create_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(128, 11, activation='swish', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(3)(x)
    x = SpatialDropout1D(0.3)(x)

    x = Conv1D(256, 7, activation='swish', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.4)(x)

    x = Conv1D(512, 5, activation='swish', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    x = Reshape((1, x.shape[1]))(x)

    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = attention_layer(x)
    x = Bidirectional(LSTM(64, kernel_regularizer=l2(1e-4)))(x)

    x = Dense(256, activation='swish', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='swish', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

model = create_model(X_train[0].shape)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, mode='min', verbose=1),
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_auc', mode='max'),
    TensorBoard(log_dir='./logs', histogram_freq=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

test_results = model.evaluate(X_test, y_test, verbose=0)

val_precision = history.history['val_precision']
best_val_precision = max(val_precision)
best_epoch = val_precision.index(best_val_precision) + 1

print(f"\nPrecision: {best_val_precision*100:.2f}%")

model.save("speech_distress_model.h5")
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)