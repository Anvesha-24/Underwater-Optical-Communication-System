import os
import cv2
import numpy as np
import random
from sequence_generator import generate_sequence

# =====================================================
# UPDATED: now generates sequences with BOTH bleed-through
# AND motion, so the CNN-LSTM learns to handle both during
# training. Each sequence gets a randomly sampled
# motion_strength, giving the model exposure to the full
# range of motion conditions.
# =====================================================

SEQUENCE_LENGTH = 20


def random_bitstring(length):
    return "".join(random.choice("01") for _ in range(length))


def save_sequence(frames, bitstring, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(folder_path, f"frame_{i:02d}.jpg"), frame)

    with open(os.path.join(folder_path, "labels.txt"), "w") as f:
        f.write(bitstring)


def generate_sequence_dataset(difficulty, num_train=2000, num_test=400,
                               motion_range=(0.0, 2.0)):
    """
    Generates labeled sequence datasets with both bleed-through
    and motion augmentation.

    motion_range: (min, max) motion_strength sampled per sequence.
    Default 0.0-2.0 covers no motion through severe motion, so
    the model sees all conditions during training.
    """
    print(f"\n=== Generating sequence dataset: {difficulty} "
          f"(motion range: {motion_range}) ===")

    train_dir = f"dataset_seq/{difficulty}/train"
    test_dir  = f"dataset_seq/{difficulty}/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Generating {num_train} training sequences...")
    for i in range(num_train):
        bitstring     = random_bitstring(SEQUENCE_LENGTH)
        # randomly sample a motion strength for this sequence
        motion_strength = np.random.uniform(*motion_range)
        frames        = generate_sequence(bitstring, difficulty,
                                          motion_strength=motion_strength)
        save_sequence(frames, bitstring,
                      os.path.join(train_dir, f"seq_{i:04d}"))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_train} done")

    print(f"Generating {num_test} test sequences...")
    for i in range(num_test):
        bitstring       = random_bitstring(SEQUENCE_LENGTH)
        motion_strength = np.random.uniform(*motion_range)
        frames          = generate_sequence(bitstring, difficulty,
                                            motion_strength=motion_strength)
        save_sequence(frames, bitstring,
                      os.path.join(test_dir, f"seq_{i:04d}"))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{num_test} done")

    print(f"'{difficulty}' sequence dataset complete: "
          f"{num_train} train + {num_test} test sequences "
          f"({SEQUENCE_LENGTH} bits each, motion range {motion_range})")


if __name__ == "__main__":
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    generate_sequence_dataset(difficulty)