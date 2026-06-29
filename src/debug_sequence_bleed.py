"""
debug_sequence_bleed.py

WHY THIS FILE?
sequence_generator.py's built-in test showed two issues:
  1. At easy/medium/hard, bit=0 frames immediately after a bit=1
     frame don't show elevated MAX brightness - need to check if
     bleed-through shows up in MEAN brightness instead (the bleed
     circle is at a different random position than the main
     signal, so it may not affect the single brightest pixel).
  2. At extreme/brutal, EVERY frame has identical brightness
     regardless of bit value - signal is being destroyed by the
     noise pipeline itself, unrelated to bleed-through.

This script checks both separately.
"""

import numpy as np
import cv2
from sequence_generator import generate_sequence, BLEED_STRENGTH_BY_DIFFICULTY
from camera_noise import add_camera_noise_at_level

# =====================================================
# PART 1: mean brightness comparison (easy/medium/hard only)
# Compare: isolated OFF (preceded by OFF) vs bleed OFF (preceded
# by ON) vs ON frame - using MEAN brightness, not max.
# =====================================================

print("=== PART 1: Mean brightness - isolated OFF vs bleed OFF vs ON ===\n")

test_bits = "1001000100"  # OFF at index 1 (after ON) and index 4 (isolated, far from any ON)

for difficulty in ["easy", "medium", "hard"]:
    frames = generate_sequence(test_bits, difficulty)

    print(f"--- {difficulty} (bleed_strength={BLEED_STRENGTH_BY_DIFFICULTY[difficulty]}) ---")
    for i, (bit, frame) in enumerate(zip(test_bits, frames)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        max_b = int(np.max(gray))
        tag = ""
        if i > 0 and test_bits[i-1] == '1' and bit == '0':
            tag = "  <- OFF right after ON (should show bleed)"
        print(f"  bit[{i}]={bit}  mean={mean_b:.2f}  max={max_b}{tag}")
    print()

# =====================================================
# PART 2: why extreme/brutal collapse to constant brightness
# Check the noise function's output at each STAGE, for both an
# all-black input and a bright-circle input, to see where the
# signal gets destroyed.
# =====================================================

print("\n=== PART 2: Where does extreme/brutal lose the signal? ===\n")

# fully black frame (simulates bit=0, no bleed)
black_frame = np.zeros((64, 64, 3), dtype=np.uint8)

# frame with a bright circle (simulates bit=1)
bright_frame = np.zeros((64, 64, 3), dtype=np.uint8)
cv2.circle(bright_frame, (32, 32), 12, (255, 255, 255), -1)

for difficulty in ["extreme", "brutal"]:
    noisy_black = add_camera_noise_at_level(black_frame.copy(), difficulty)
    noisy_bright = add_camera_noise_at_level(bright_frame.copy(), difficulty)

    gray_black = cv2.cvtColor(noisy_black, cv2.COLOR_BGR2GRAY)
    gray_bright = cv2.cvtColor(noisy_bright, cv2.COLOR_BGR2GRAY)

    print(f"--- {difficulty} ---")
    print(f"  BLACK input  -> max={np.max(gray_black)}  mean={np.mean(gray_black):.2f}")
    print(f"  BRIGHT input -> max={np.max(gray_bright)}  mean={np.mean(gray_bright):.2f}")
    print(f"  Difference in max brightness: {int(np.max(gray_bright)) - int(np.max(gray_black))}")
    print()