import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# ===============================
# STEP 1 – SET DATA PATHS
# ===============================
train_dir = r"C:\FINAL\FL_BrainTumor\data\Training"
test_dir  = r"C:\FINAL\FL_BrainTumor\data\Testing"

IMAGE_SIZE = 224
BATCH_SIZE = 32

# ===============================
# STEP 2 – DATA AUGMENTATION
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ===============================
# STEP 3 – BUILD MODEL
# ===============================
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

base.trainable = False   # freeze backbone at first

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.summary()

# ===============================
# STEP 4 – TRAIN MODEL (PHASE 1)
# ===============================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=8
)

# Unfreeze base model for fine-tuning
base.trainable = True
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=['accuracy']
)

# ===============================
# STEP 5 – TRAIN MODEL (PHASE 2)
# ===============================
history2 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=8
)

# ===============================
# STEP 6 – SAVE MODEL (.h5)
# ===============================
model.save("brain_tumor_model.h5")
print("Model saved as brain_tumor_model.h5")

# ===============================
# STEP 7 – EXPORT TFLITE MODEL
# ===============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("brain_tumor_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as brain_tumor_model.tflite")
