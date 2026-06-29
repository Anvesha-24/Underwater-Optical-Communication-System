"""
trace_noise_stages.py

WHY THIS FILE?
debug_bleed_averaged.py proved (over 100 trials, noise averaged
out) that a frame containing a FAINT bleed circle ends up DIMMER
on average than a frame with NOTHING in it (pure black). That's
backwards - adding any brightness to a black frame should never
make it end up dimmer after processing. This script manually
walks through each stage inside add_camera_noise() for both
cases, to find exactly where the reversal happens.
"""

import numpy as np
import cv2

IMG_SIZE = 64

def trace_stages(img, label):
    print(f"\n--- {label} ---")

    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  stage 0 (input):       mean={np.mean(gray0):.3f}  max={np.max(gray0)}")

    # stage 1: red channel suppression
    b, g, r = cv2.split(img)
    r = (r * 0.3).astype(np.uint8)
    img1 = cv2.merge((b, g, r))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print(f"  stage 1 (red supp.):   mean={np.mean(gray1):.3f}  max={np.max(gray1)}")

    # stage 2: blur
    blur_size = 9  # easy
    img2 = cv2.GaussianBlur(img1, (blur_size, blur_size), 0)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(f"  stage 2 (blur):        mean={np.mean(gray2):.3f}  max={np.max(gray2)}")

    # stage 3: darken
    img3 = cv2.convertScaleAbs(img2, alpha=0.85, beta=-10)  # easy preset
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    print(f"  stage 3 (darken):      mean={np.mean(gray3):.3f}  max={np.max(gray3)}")

    # stage 4: noise (use a FIXED seed so both runs get identical
    # random noise, isolating the effect of the input itself)
    np.random.seed(42)
    noise = np.random.randint(0, 15, img3.shape, dtype='uint8')
    img4 = cv2.addWeighted(img3, 0.9, noise, 0.1, 0)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    print(f"  stage 4 (noise):       mean={np.mean(gray4):.3f}  max={np.max(gray4)}")

    return gray4


# Case A: pure black frame (isolated OFF)
black = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# Case B: faded bleed circle on black background (bleed-OFF)
# matches sequence_generator's bleed logic: take a circle layer,
# multiply by bleed_strength (0.05 for easy), add to blank base
bleed_strength = 0.05
circle_layer = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
cv2.circle(circle_layer, (32, 32), 10, (230, 230, 230), -1)
faded_bleed = (circle_layer.astype(np.float32) * bleed_strength).astype(np.uint8)
bleed_input = cv2.add(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), faded_bleed)

trace_stages(black, "Case A: pure black (no bleed)")
trace_stages(bleed_input, "Case B: faded bleed circle on black")