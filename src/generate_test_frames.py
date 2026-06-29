"""
generate_test_frames.py

Generates a set of sample frames and saves them to
outputs/test_frames/ so you can test the "Upload my own frames"
feature in the Receiver tab of the Streamlit app.

Run from src/:
    python generate_test_frames.py

Then in the app:
  1. Go to Receiver tab
  2. Select "Upload my own frames"
  3. Upload all files from outputs/test_frames/
  4. Paste the printed bitstring into "True bitstring" field
  5. Click Decode
"""

import cv2
import os
from encoder import text_to_binary
from sequence_generator import generate_sequence

# =====================================================
# SETTINGS - change these to test different conditions
# =====================================================
MESSAGE    = "OK"
DIFFICULTY = "medium"
MOTION     = 1.5
# =====================================================

os.makedirs("outputs/test_frames", exist_ok=True)

binary = text_to_binary(MESSAGE)
frames = generate_sequence(binary, DIFFICULTY, motion_strength=MOTION)

for i, frame in enumerate(frames):
    path = f"outputs/test_frames/frame_{i:02d}.jpg"
    cv2.imwrite(path, frame)

print(f"Message   : {MESSAGE}")
print(f"Bitstring : {binary}")
print(f"Difficulty: {DIFFICULTY}, Motion: {MOTION}")
print(f"Saved {len(frames)} frames to outputs/test_frames/")
print(f"\nIn the app:")
print(f"  1. Go to Receiver tab")
print(f"  2. Select 'Upload my own frames'")
print(f"  3. Upload all files from outputs/test_frames/")
print(f"  4. Paste this bitstring into 'True bitstring': {binary}")
print(f"  5. Click Decode Frames")