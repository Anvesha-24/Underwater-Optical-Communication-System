"""
check_on_real_training_images.py

WHY THIS FILE?
check_class_labels.py confirmed labels are correct (off=0, on=1), but
debug_signals.py still showed the CNN scoring clearly-brighter "ON"
frames as near-0 probability. That means either:

  (a) the model is broken / not loading correctly, OR
  (b) the newly generated frames (in bit_to_frame) don't match the
      brightness/appearance of the images the CNN actually trained on

This script loads a few REAL images straight from dataset/train/on and
dataset/train/off (the exact files used during training) and runs them
through the CNN. If the CNN scores these correctly (high prob for "on"
images, low for "off"), the model itself is fine, and the bug is in
how bit_to_frame() generates frames for the live pipeline. If the CNN
gets even these wrong, the model/loading is the problem.

Run this in src/.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 64
model = load_model("flashlight_cnn.h5")


def load_and_predict(folder, label, num_samples=5):
    files = sorted(os.listdir(folder))[:num_samples]

    print(f"\n--- {label} images from {folder} ---")
    print(f"{'file':<15}{'max_brightness':<16}{'mean_brightness':<18}{'cnn_prob':<10}")

    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)

        if img is None:
            print(f"{fname:<15} could not read image")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_val = int(np.max(gray))
        mean_val = float(np.mean(gray))

        # same preprocessing as training: resize (should already be 64x64,
        # resizing anyway in case it isn't) + rescale to 0-1
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cnn_input = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
        prob = model.predict(cnn_input, verbose=0)[0][0]

        print(f"{fname:<15}{max_val:<16}{mean_val:<18.2f}{prob:<10.4f}")


load_and_predict("dataset/train/on", "ON")
load_and_predict("dataset/train/off", "OFF")