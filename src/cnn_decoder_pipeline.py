"""
cnn_decoder_pipeline.py

Connects the trained flashlight CNN (from train_cnn.py) into the
end-to-end bit-recovery pipeline, and compares it against the
fixed-threshold and adaptive-threshold detectors using BER.

FIX (from debugging session):
OpenCV loads/creates images in BGR channel order, but Keras's
flow_from_directory (used during training) loads images via PIL in
RGB order. Since camera_noise.py specifically suppresses the red
channel (r = r * 0.3, simulating underwater red-light absorption),
BGR vs RGB is NOT interchangeable here - feeding the CNN a BGR array
makes it see the channels in the wrong order, which was causing
near-random predictions. Every frame must be converted BGR->RGB
before being passed to model.predict().

Place this file in your `src/` folder, alongside encoder.py,
decoder.py, camera_noise.py, threshold_detector.py,
adaptive_threshold.py, ber_calculation.py and flashlight_cnn.h5
(the model saved by train_cnn.py).
"""

import random
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from encoder import text_to_binary
from decoder import binary_to_text
from camera_noise import add_camera_noise
from threshold_detector import signal_to_binary
from adaptive_threshold import adaptive_signal_to_binary
from ber_calculation import calculate_ber


IMG_SIZE = 64
model = load_model("flashlight_cnn.h5")


# =====================================================
# Render one bit as a noisy 64x64 frame, same style as
# dataset_generator.py's generate_on_image / generate_off_image
# (OpenCV functions like cv2.circle naturally produce BGR images,
# since that's OpenCV's default - this is fine for camera_noise.py,
# the conversion to RGB happens later, only right before the CNN)
# =====================================================
def bit_to_frame(bit):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    if bit == '1':
        x = np.random.randint(20, 44)
        y = np.random.randint(20, 44)
        radius = np.random.randint(8, 13)
        brightness = np.random.randint(220, 256)
        cv2.circle(img, (x, y), radius, (brightness,) * 3, -1)

    return add_camera_noise(img)


# =====================================================
# Reduce a noisy frame to a single brightness value, so the
# fixed/adaptive threshold detectors (which work on 1D signals)
# can be compared fairly against the CNN on the same frames.
# Channel order doesn't matter here since we're just taking max
# grayscale brightness, not feeding this into the CNN.
# =====================================================
def frame_to_signal_value(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return int(np.max(gray))


# =====================================================
# CNN decoding for a whole batch of frames at once
# (class 0 = "off", class 1 = "on", confirmed via
# check_class_labels.py)
#
# THE FIX: convert each BGR frame to RGB before stacking into
# a batch, so the CNN sees the same channel order it trained on.
# =====================================================
def cnn_decode(frames):
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    batch = np.array(rgb_frames, dtype="float32") / 255.0
    preds = model.predict(batch, verbose=0)
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)


# =====================================================
# Run one message through all three decoders
# =====================================================
def run_demo(message="HELLO"):
    binary = text_to_binary(message)

    frames = [bit_to_frame(b) for b in binary]
    signal_values = [frame_to_signal_value(f) for f in frames]

    fixed_bits = signal_to_binary(signal_values)
    adaptive_bits, threshold_used = adaptive_signal_to_binary(signal_values)
    cnn_bits = cnn_decode(frames)

    print("Original :", binary)
    print("Fixed    :", fixed_bits)
    print("Adaptive :", adaptive_bits, "(threshold:", round(threshold_used, 1), ")")
    print("CNN      :", cnn_bits)

    for name, bits in [("Fixed", fixed_bits), ("Adaptive", adaptive_bits), ("CNN", cnn_bits)]:
        errors, ber = calculate_ber(binary, bits)
        print(f"{name:<8} -> errors: {errors}/{len(binary)}, BER: {ber:.3f}")

    print("\nCNN decoded message:", binary_to_text(cnn_bits))


# =====================================================
# Run many random bitstreams and average the BER, so the
# comparison isn't based on one lucky/unlucky message
# =====================================================
def run_ber_experiment(num_bits=200, trials=10):
    totals = {"Fixed": 0.0, "Adaptive": 0.0, "CNN": 0.0}

    for _ in range(trials):
        binary = "".join(random.choice("01") for _ in range(num_bits))

        frames = [bit_to_frame(b) for b in binary]
        signal_values = [frame_to_signal_value(f) for f in frames]

        fixed_bits = signal_to_binary(signal_values)
        adaptive_bits, _ = adaptive_signal_to_binary(signal_values)
        cnn_bits = cnn_decode(frames)

        for name, bits in [("Fixed", fixed_bits), ("Adaptive", adaptive_bits), ("CNN", cnn_bits)]:
            _, ber = calculate_ber(binary, bits)
            totals[name] += ber

    print(f"\nAverage BER over {trials} trials of {num_bits} bits each:")
    for name, total in totals.items():
        print(f"  {name:<8}: {total / trials:.4f}")


if __name__ == "__main__":
    run_demo("HELLO")
    run_ber_experiment()