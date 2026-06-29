"""
check_weak_on_images.py

WHY THIS FILE?
After retraining with more epochs, ON accuracy is still only 110/200
(~55%), while OFF is 200/200. Since OFF images are always identical
(plain black), but ON images have randomized circle position/radius
(5-12px)/brightness(180-255), the theory is: after add_camera_noise's
blur+darken+noise, WEAK on-images (small radius, low brightness) end
up looking almost identical to OFF images - so the model is actually
behaving reasonably, the data itself is ambiguous for those cases.

This script sorts ON test images by their post-noise max brightness
and shows the CNN's prediction for the dimmest vs brightest ones, to
confirm whether failures cluster among the dim/weak examples.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 64
model = load_model("flashlight_cnn.h5")

folder = "dataset/test/on"
files = sorted(os.listdir(folder))[:200]

results = []

for fname in files:
    path = os.path.join(folder, fname)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_brightness = int(np.max(gray))

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cnn_input = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
    prob = model.predict(cnn_input, verbose=0)[0][0]

    results.append((fname, max_brightness, prob))

# sort by brightness, dimmest first
results.sort(key=lambda r: r[1])

print(f"{'file':<15}{'max_brightness':<16}{'cnn_prob':<10}{'correct?':<10}")
print("-" * 55)

print("\n--- 10 DIMMEST on-images ---")
for fname, brightness, prob in results[:10]:
    correct = "YES" if prob > 0.5 else "NO"
    print(f"{fname:<15}{brightness:<16}{prob:<10.4f}{correct:<10}")

print("\n--- 10 BRIGHTEST on-images ---")
for fname, brightness, prob in results[-10:]:
    correct = "YES" if prob > 0.5 else "NO"
    print(f"{fname:<15}{brightness:<16}{prob:<10.4f}{correct:<10}")

# overall correlation check
brightness_vals = [r[1] for r in results]
correct_flags = [1 if r[2] > 0.5 else 0 for r in results]

print(f"\nMean brightness of CORRECT predictions: "
      f"{np.mean([b for b, c in zip(brightness_vals, correct_flags) if c == 1]):.2f}")
print(f"Mean brightness of WRONG predictions:   "
      f"{np.mean([b for b, c in zip(brightness_vals, correct_flags) if c == 0]):.2f}")