"""
compare_pil_vs_cv2_loading.py

WHY THIS FILE?
train_cnn.py's built-in validation reports val_accuracy=1.0000 on
dataset/test, but our manual check scripts (using cv2.imread) get
only 38/200 correct on the SAME folder. Since both can't be true on
identical data, the most likely explanation is a preprocessing
mismatch: Keras's flow_from_directory loads images via PIL (RGB
channel order, PIL's resize), while our debug scripts use OpenCV
(BGR channel order, cv2's resize).

This script loads the same handful of files BOTH ways and compares:
  - raw pixel/brightness stats
  - the CNN's prediction on each version

If the PIL-loaded version scores correctly but the OpenCV-loaded
version doesn't, that confirms the mismatch and tells us exactly
what to fix in our debug scripts (and in cnn_decoder_pipeline.py,
which also uses cv2 - meaning the live pipeline could have this same
bug).
"""

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 64
model = load_model("flashlight_cnn.h5")

test_files = [
    "dataset/test/on/on_0.jpg",
    "dataset/test/on/on_1.jpg",
    "dataset/test/on/on_2.jpg",
    "dataset/test/on/on_3.jpg",
    "dataset/test/on/on_4.jpg",
]

print(f"{'file':<25}{'method':<8}{'max_brightness':<16}{'cnn_prob':<10}")
print("-" * 60)

for path in test_files:

    # ---- Method 1: PIL, same as Keras's flow_from_directory ----
    pil_img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    pil_array = img_to_array(pil_img)  # RGB order, float32, 0-255

    gray_pil = np.mean(pil_array, axis=2)  # quick grayscale approx
    max_pil = int(np.max(gray_pil))

    pil_input = np.expand_dims(pil_array / 255.0, axis=0)
    prob_pil = model.predict(pil_input, verbose=0)[0][0]

    print(f"{path:<25}{'PIL':<8}{max_pil:<16}{prob_pil:<10.4f}")

    # ---- Method 2: OpenCV, same as our debug scripts ----
    cv_img = cv2.imread(path)  # BGR order, uint8
    cv_img_resized = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE))

    gray_cv = cv2.cvtColor(cv_img_resized, cv2.COLOR_BGR2GRAY)
    max_cv = int(np.max(gray_cv))

    cv_input = np.expand_dims(cv_img_resized.astype("float32") / 255.0, axis=0)
    prob_cv = model.predict(cv_input, verbose=0)[0][0]

    print(f"{path:<25}{'OpenCV':<8}{max_cv:<16}{prob_cv:<10.4f}")
    print()