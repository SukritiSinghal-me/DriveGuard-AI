import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

# ==========================================================
# GPU Configuration
# ==========================================================

gpus = tf.config.list_physical_devices('GPU')

if len(gpus) > 0:
    print("\nGPU Found!")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

else:
    print("\nRunning on CPU")

# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/validate"
TEST_DIR = "dataset/test"

# ==========================================================
# Image Parameters
# ==========================================================

IMG_SIZE = 128

BATCH_SIZE = 64

EPOCHS = 2

# ==========================================================
# Data Generator
# ==========================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True

)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================================================
# Data Loaders
# ==========================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True

)

validation_generator = test_datagen.flow_from_directory(

    VALID_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary"

)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

# ==========================================================
# CNN Model
# ==========================================================

model = Sequential([

    Conv2D(32, (3,3), activation='relu',
           input_shape=(128,128,3)),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Conv2D(256,(3,3),activation='relu'),

    BatchNormalization(),

    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),

    Dense(128,activation='relu'),

    Dropout(0.4),

    Dense(1,activation='sigmoid',dtype='float32')

])

# ==========================================================
# Compile
# ==========================================================

model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ==========================================================
# Callbacks
# ==========================================================

checkpoint = ModelCheckpoint(

    "models/best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    verbose=1

)

# ==========================================================
# Training
# ==========================================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[checkpoint, early_stop, reduce_lr]

)

# ==========================================================
# Testing
# ==========================================================

loss, accuracy = model.evaluate(test_generator)

print("\n========================")
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)
print("========================")

# ==========================================================
# Save Final Model
# ==========================================================

model.save("models/driver_drowsiness_model.keras")

print("\nModel Saved Successfully!")

# ==========================================================
# Accuracy Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.savefig("models/accuracy.png")

plt.close()

# ==========================================================
# Loss Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.savefig("models/loss.png")

plt.close()

print("\nTraining Completed Successfully!")