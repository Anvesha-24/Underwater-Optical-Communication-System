import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import cv2

# =====================================================
# DATASET PATH
# =====================================================

train_dir = "dataset/train"
test_dir = "dataset/test"

# =====================================================
# IMAGE PREPROCESSING
#
# WHY shuffle + seed?
# Earlier training run got 100% reported accuracy but the
# model had a blind spot on harder/dimmer "on" images.
# Fixing the seed makes results reproducible while we debug,
# and explicit shuffling avoids any ordering effects between
# the "on" and "off" folders.
# =====================================================

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # keep test order fixed, makes debugging easier later
)

print("Class indices:", train_generator.class_indices)

# =====================================================
# CNN MODEL
# (unchanged architecture - the issue was training, not the model shape)
# =====================================================

model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# TRAIN MODEL
#
# WHY MORE EPOCHS + EARLY STOPPING?
# 5 epochs wasn't enough for the model to learn the dimmer/
# harder "on" examples - it settled into a shortcut of mostly
# predicting "off" while still getting low overall loss (since
# it nailed all "off" images and some easy "on" images).
# More epochs give it a chance to actually learn the harder
# cases. EarlyStopping prevents wasting time/overfitting once
# validation loss stops improving.
# =====================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,
    callbacks=[early_stop]
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("flashlight_cnn.h5")
print("Model saved")

# =====================================================
# PER-CLASS ACCURACY CHECK
#
# WHY?
# Overall accuracy/loss can look perfect while hiding a
# blind spot on one class (exactly what happened before).
# This explicitly checks "on" and "off" images separately
# so any imbalance is caught immediately, not discovered
# later through a confusing downstream BER bug.
# =====================================================

def check_class_accuracy(folder, label, num_samples=200):
    files = sorted(os.listdir(folder))[:num_samples]
    images = []

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        img = cv2.resize(img, (64, 64))
        images.append(img.astype("float32") / 255.0)

    batch = np.array(images)
    preds = model.predict(batch, verbose=0).flatten()

    if label == "on":
        correct = np.sum(preds > 0.5)
    else:
        correct = np.sum(preds < 0.5)

    print(f"{label.upper()} accuracy: {correct}/{len(preds)}  "
          f"(mean prob: {preds.mean():.4f})")


print("\n--- Per-class accuracy on test set ---")
check_class_accuracy("dataset/test/on", "on")
check_class_accuracy("dataset/test/off", "off")