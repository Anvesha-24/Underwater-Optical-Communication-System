import cv2
import numpy as np
import os
from camera_noise import add_camera_noise_at_level, DIFFICULTY_PRESETS

# =====================================================
# WHY SEPARATE FOLDERS PER DIFFICULTY?
# We need one CNN trained per difficulty level, and a fair
# BER comparison at each level. Keeping datasets fully
# separate (dataset/easy/, dataset/medium/, etc.) means each
# training run and each experiment is unambiguous about
# which difficulty it's using.
# =====================================================

def make_folders(difficulty):
    folders = [
        f"dataset/{difficulty}/train/on",
        f"dataset/{difficulty}/train/off",
        f"dataset/{difficulty}/test/on",
        f"dataset/{difficulty}/test/off",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return folders


# =====================================================
# GENERATE CLEAN FLASHLIGHT IMAGE (before noise)
# Same randomized circle as before - position/size/brightness
# vary, but stay within a range that's learnable at "easy"
# difficulty. Harder difficulties apply harsher noise on TOP
# of this same base image, which is what actually creates
# the difficulty difference.
# =====================================================

def generate_on_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    x = np.random.randint(20, 44)
    y = np.random.randint(20, 44)
    radius = np.random.randint(8, 13)
    brightness = np.random.randint(220, 256)

    cv2.circle(img, (x, y), radius, (brightness, brightness, brightness), -1)
    return img


def generate_off_image():
    return np.zeros((64, 64, 3), dtype=np.uint8)


# =====================================================
# SAFETY CHECK: regenerate if noise washed out the signal
#
# WHY min_brightness IS NOW DIFFICULTY-DEPENDENT?
# At "easy" difficulty, we can demand a high minimum
# brightness (e.g. 140) since the noise is mild. At "extreme"
# difficulty, the darkening (alpha=0.40, beta=-40) makes even
# a perfect bright circle end up dim - demanding 140 there
# would retry forever and never succeed. So each difficulty
# gets its own realistic minimum, low enough to be achievable,
# but still above the OFF-image noise floor for that level.
# =====================================================

MIN_BRIGHTNESS_BY_DIFFICULTY = {
    "easy": 140,
    "medium": 100,
    "hard": 65,
    "extreme": 40,
}


def generate_valid_on_image(difficulty, max_attempts=15):
    min_brightness = MIN_BRIGHTNESS_BY_DIFFICULTY[difficulty]

    last_attempt = None
    for _ in range(max_attempts):
        img = generate_on_image()
        noisy_img = add_camera_noise_at_level(img, difficulty)
        last_attempt = noisy_img

        gray = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
        if np.max(gray) >= min_brightness:
            return noisy_img

    # if every attempt fell short, just return the last one -
    # rare, and prevents an infinite loop
    return last_attempt


# =====================================================
# "brutal" gets its OWN generation function with NO safety
# check / regeneration filter.
#
# WHY?
# Every other difficulty uses generate_valid_on_image(), which
# retries until it finds a clean, comfortably-bright image -
# this guarantees the dataset never contains genuinely
# ambiguous/unrecoverable signal, which is why even "extreme"
# came out at a flat 0.0000 BER for the CNN. To find the CNN's
# real breaking point, "brutal" images are saved as-is, however
# washed-out they happen to come out - some will be barely
# distinguishable from OFF, which is the whole point.
# =====================================================

def generate_brutal_on_image_unfiltered():
    img = generate_on_image()
    return add_camera_noise_at_level(img, "brutal")


# =====================================================
# SAVE ONE DIFFICULTY LEVEL'S DATASET
# =====================================================

def create_dataset_for_difficulty(difficulty, train_count=5000, test_count=1000):
    print(f"\n=== Generating '{difficulty}' dataset ===")

    on_train_dir = f"dataset/{difficulty}/train/on"
    off_train_dir = f"dataset/{difficulty}/train/off"
    on_test_dir = f"dataset/{difficulty}/test/on"
    off_test_dir = f"dataset/{difficulty}/test/off"

    # "brutal" uses the unfiltered generator (no safety-check
    # retry), every other difficulty uses the filtered one
    if difficulty == "brutal":
        on_image_fn = lambda: generate_brutal_on_image_unfiltered()
    else:
        on_image_fn = lambda: generate_valid_on_image(difficulty)

    print(f"Generating {difficulty} TRAIN set...")

    for i in range(train_count):
        img = on_image_fn()
        cv2.imwrite(f"{on_train_dir}/on_{i}.jpg", img)

    for i in range(train_count):
        img = generate_off_image()
        img = add_camera_noise_at_level(img, difficulty)
        cv2.imwrite(f"{off_train_dir}/off_{i}.jpg", img)

    print(f"Generating {difficulty} TEST set...")

    for i in range(test_count):
        img = on_image_fn()
        cv2.imwrite(f"{on_test_dir}/on_{i}.jpg", img)

    for i in range(test_count):
        img = generate_off_image()
        img = add_camera_noise_at_level(img, difficulty)
        cv2.imwrite(f"{off_test_dir}/off_{i}.jpg", img)

    print(f"'{difficulty}' dataset complete")


# =====================================================
# MAIN - generate datasets for ALL difficulty levels
# =====================================================

def create_all_datasets():
    for difficulty in DIFFICULTY_PRESETS.keys():
        make_folders(difficulty)
        create_dataset_for_difficulty(difficulty)

    print("\nAll difficulty datasets generated.")


def create_single_dataset(difficulty):
    """Generate just one difficulty level - useful when you've
    already generated the others and only added a new preset
    (like 'brutal'), so you don't have to redo everything."""
    make_folders(difficulty)
    create_dataset_for_difficulty(difficulty)
    print(f"\n'{difficulty}' dataset generated.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # e.g. python dataset_generator.py brutal
        create_single_dataset(sys.argv[1])
    else:
        create_all_datasets()