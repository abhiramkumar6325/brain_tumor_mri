# convert_model.py
import tensorflow as tf
import os

MODEL_DIR = "models"
keras_model_path = os.path.join(MODEL_DIR, "model.h5")
tflite_path = os.path.join(MODEL_DIR, "model.tflite")

if not os.path.exists(keras_model_path):
    raise SystemExit("model.h5 not found. Train the model first.")

print("Loading Keras model...")
model = tf.keras.models.load_model(keras_model_path)

# -------------------------------------------
# 🔥 COMPATIBLE TFLITE CONVERTER SETTINGS
# -------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Use OLD converter (Flutter compatible)
converter.experimental_new_converter = False

# Use only TFLITE_BUILTINS (no Flex ops)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Disable per-channel quantization (Flutter cannot load)
converter._experimental_disable_per_channel = True

print("Converting to TFLite...")
tflite_model = converter.convert()

os.makedirs(MODEL_DIR, exist_ok=True)
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("✔ TFLite model saved → models/model.tflite")
print("✔ Compatible with Flutter tflite_flutter 0.10.0")
