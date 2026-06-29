"""
debug_bleed_averaged.py

WHY THIS FILE?
A single run of sequence_generator showed bleed-OFF frames (after
a '1') with LOWER brightness than isolated OFF frames - backwards
from what bleed-through should do. Since add_camera_noise() adds a
random noise term on every call, a single sample can easily be
dominated by that randomness rather than showing the real
(possibly much smaller) bleed effect. This script averages over
many trials to filter out noise and reveal the true effect.
"""

import numpy as np
import cv2
from sequence_generator import generate_sequence

def average_brightness_by_position(test_bits, difficulty, trials=100):
    """Runs generate_sequence() many times and averages the mean
    brightness at each bit position, to cancel out per-call random
    noise and reveal the true underlying signal (including bleed)."""

    num_positions = len(test_bits)
    totals = np.zeros(num_positions)

    for _ in range(trials):
        frames = generate_sequence(test_bits, difficulty)
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            totals[i] += np.mean(gray)

    return totals / trials


test_bits = "1001000100"

for difficulty in ["easy", "medium", "hard"]:
    print(f"\n--- {difficulty} (averaged over 100 trials) ---")
    avg_brightness = average_brightness_by_position(test_bits, difficulty, trials=100)

    for i, bit in enumerate(test_bits):
        tag = ""
        if i > 0 and test_bits[i-1] == '1' and bit == '0':
            tag = "  <- OFF right after ON (should show bleed)"
        print(f"  bit[{i}]={bit}  avg_mean_brightness={avg_brightness[i]:.3f}{tag}")