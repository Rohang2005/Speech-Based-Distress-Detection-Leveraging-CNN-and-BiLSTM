import tensorflow as tf
from tensorflow.keras import backend as K

# Define focal loss again
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * K.pow(1. - pt, gamma) * K.log(pt)
        return K.mean(loss)
    return focal_loss_fixed

model = tf.keras.models.load_model(
    "speech_distress_model.h5",
    custom_objects={'focal_loss_fixed': focal_loss()}
)

print("Model loaded successfully.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_enable_resource_variables = True
converter._experimental_lower_tensor_list_ops = False
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("speech_distress_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model  saved as 'speech_distress_model.tflite'")
