"""
ber_cnn_vs_cnn_lstm.py

Compares the plain single-frame CNN (flashlight_cnn_medium.h5)
against the CNN-LSTM sequence decoder (cnn_lstm_medium.h5)
on sequences with inter-symbol interference (ISI/bleed-through).

WHY THIS EXPERIMENT?
The single-frame CNN was already perfect at classifying individual
frames independently. But real underwater scattering causes light
from one bit to bleed into the next frame - a problem a
frame-by-frame classifier can't handle because it has no memory
of what the previous frame looked like. This experiment directly
tests whether the CNN-LSTM's sequence awareness actually helps
by sweeping bleed_strength from 0 (no ISI) to high values, and
measuring BER for both models at each level.

Expected result: at bleed_strength=0 both models should perform
similarly (no ISI to exploit). As bleed_strength increases, the
plain CNN should degrade faster than the CNN-LSTM.
"""

import numpy as np
import cv2
import random
from tensorflow.keras.models import load_model
from sequence_generator import generate_sequence, BLEED_STRENGTH_BY_DIFFICULTY
from camera_noise import add_camera_noise_at_level

IMG_SIZE = 64
SEQUENCE_LENGTH = 20
DIFFICULTY = "medium"
NUM_TRIALS = 50  # sequences per bleed level - enough to get stable BER

# load both models once
print("Loading models...")
cnn_model = load_model(f"flashlight_cnn_{DIFFICULTY}.h5")
lstm_model = load_model(f"cnn_lstm_{DIFFICULTY}.h5")
print("Models loaded.")


def random_bitstring(length):
    return "".join(random.choice("01") for _ in range(length))


# =====================================================
# GENERATE SEQUENCE WITH CUSTOM BLEED STRENGTH
#
# We want to test a range of bleed strengths, not just the
# preset for "medium" - so we replicate sequence_generator's
# logic here but with an overridable bleed_strength parameter.
# =====================================================

def generate_sequence_with_bleed(bitstring, difficulty, bleed_strength):
    """Same as sequence_generator.generate_sequence() but with
    bleed_strength as an explicit parameter instead of looking
    it up from the preset - so we can sweep it independently."""

    frames = []
    previous_circle_layer = None

    for bit in bitstring:
        base = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if previous_circle_layer is not None:
            bleed = (previous_circle_layer.astype(np.float32) * bleed_strength).astype(np.uint8)
            base = cv2.add(base, bleed)

        current_circle_layer = None

        if bit == '1':
            x = np.random.randint(20, 44)
            y = np.random.randint(20, 44)
            radius = np.random.randint(8, 13)
            brightness = np.random.randint(220, 256)

            circle_layer = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            cv2.circle(circle_layer, (x, y), radius, (brightness,) * 3, -1)
            current_circle_layer = circle_layer
            base = cv2.add(base, circle_layer)

        noisy_frame = add_camera_noise_at_level(base, difficulty)
        frames.append(noisy_frame)
        previous_circle_layer = current_circle_layer

    return frames


# =====================================================
# PLAIN CNN DECODING (frame by frame, no sequence context)
# =====================================================

def decode_with_cnn(frames):
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    batch = np.array(rgb_frames, dtype="float32") / 255.0
    preds = cnn_model.predict(batch, verbose=0)
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)


# =====================================================
# CNN-LSTM DECODING (sequence-aware)
# =====================================================

def decode_with_cnn_lstm(frames):
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    seq = np.array(rgb_frames, dtype="float32") / 255.0
    seq = np.expand_dims(seq, axis=0)  # add batch dimension: (1, 20, 64, 64, 3)
    preds = lstm_model.predict(seq, verbose=0)[0]  # shape: (20, 1)
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)


# =====================================================
# BER CALCULATION
# =====================================================

def calculate_ber(original, received):
    errors = sum(a != b for a, b in zip(original, received))
    return errors / len(original)


# =====================================================
# SWEEP BLEED STRENGTH AND MEASURE BER
# =====================================================

def run_comparison():
    # sweep from 0 (no ISI) to high ISI - 0.6 is quite severe
    bleed_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print(f"\n{'='*60}")
    print("BER Comparison: Plain CNN vs CNN-LSTM under increasing ISI")
    print(f"Difficulty: {DIFFICULTY}, Trials per level: {NUM_TRIALS}")
    print(f"{'='*60}")
    print(f"{'bleed_strength':<16}{'CNN BER':<12}{'CNN-LSTM BER':<14}{'winner'}")
    print("-" * 55)

    results = {}

    for bleed in bleed_levels:
        cnn_ber_total = 0.0
        lstm_ber_total = 0.0

        for _ in range(NUM_TRIALS):
            bitstring = random_bitstring(SEQUENCE_LENGTH)
            frames = generate_sequence_with_bleed(bitstring, DIFFICULTY, bleed)

            cnn_bits = decode_with_cnn(frames)
            lstm_bits = decode_with_cnn_lstm(frames)

            cnn_ber_total += calculate_ber(bitstring, cnn_bits)
            lstm_ber_total += calculate_ber(bitstring, lstm_bits)

        cnn_avg = cnn_ber_total / NUM_TRIALS
        lstm_avg = lstm_ber_total / NUM_TRIALS

        if lstm_avg < cnn_avg:
            winner = "CNN-LSTM"
        elif cnn_avg < lstm_avg:
            winner = "CNN"
        else:
            winner = "tie"

        results[bleed] = (cnn_avg, lstm_avg)
        print(f"{bleed:<16.1f}{cnn_avg:<12.4f}{lstm_avg:<14.4f}{winner}")

    return results


if __name__ == "__main__":
    run_comparison()