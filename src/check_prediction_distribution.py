"""
check_prediction_distribution.py

WHY THIS FILE?
check_on_real_training_images.py showed the CNN scoring near-0 for
BOTH on_*.jpg and off_*.jpg images, even though their brightness is
clearly very different (ON ~75-87 max brightness, OFF ~21-22). That
means the model isn't using the obvious brightness signal at all -
it may have collapsed during training to just always predict "off".

This script runs the CNN over 100 ON images + 100 OFF images and
prints summary stats (min/max/mean probability for each class), to
confirm the collapse across a larger sample rather than just 5 files.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 64
model = load_model("flashlight_cnn.h5")


def get_predictions(folder, num_samples=100):
    files = sorted(os.listdir(folder))[:num_samples]
    images = []

    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img.astype("float32") / 255.0)

    batch = np.array(images)
    preds = model.predict(batch, verbose=0).flatten()
    return preds


on_preds = get_predictions("dataset/train/on")
off_preds = get_predictions("dataset/train/off")

print("ON images  (n={}): min={:.4f}  max={:.4f}  mean={:.4f}".format(
    len(on_preds), on_preds.min(), on_preds.max(), on_preds.mean()))

print("OFF images (n={}): min={:.4f}  max={:.4f}  mean={:.4f}".format(
    len(off_preds), off_preds.min(), off_preds.max(), off_preds.mean()))

print("\nHow many ON images predicted >0.5 (correct):", np.sum(on_preds > 0.5), "/", len(on_preds))
print("How many OFF images predicted <0.5 (correct):", np.sum(off_preds < 0.5), "/", len(off_preds))