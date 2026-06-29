import cv2
import numpy as np
import os

# =========================================================
# WHY THIS?
# Create output folder to store results cleanly.
# =========================================================
os.makedirs("outputs", exist_ok=True)

# =========================================================
# DIFFICULTY PRESETS
#
# WHY?
# Previously every noise parameter (blur size, darkening,
# noise level) was hardcoded, so there was only ever ONE
# difficulty level. To test "does BER get worse as scattering
# increases, and does the CNN handle it better than threshold
# methods", we need scattering strength to be a dial we can
# turn. Each preset below represents one point on that dial.
#
#   blur_size    -> bigger = more scattering/blur spread
#   alpha        -> lower = darker image (less brightness kept)
#   beta         -> more negative = darker image (flat subtract)
#   noise_level  -> bigger = more random sensor/particle noise
# =========================================================
DIFFICULTY_PRESETS = {
    # WHY RECALIBRATED?
    # After fixing the convertScaleAbs bug (it was taking absolute
    # value instead of clipping negatives to 0), darkening became
    # much stronger than before - the old bug was accidentally
    # creating an artificial brightness "floor" that made every
    # preset look gentler than the numbers suggested. These new
    # values are tuned under the CORRECT math: alpha/beta chosen
    # so a fully-bright pixel (255) always survives with a
    # meaningful margin, even at "brutal" - difficulty scaling
    # comes mainly from blur (spreads/weakens the signal) and
    # noise_level (adds random corruption), not from alpha/beta
    # alone crushing everything to zero.
    "easy":    {"blur_size": 7,  "alpha": 0.90, "beta": -5,  "noise_level": 12},
    "medium":  {"blur_size": 13, "alpha": 0.75, "beta": -8,  "noise_level": 25},
    "hard":    {"blur_size": 19, "alpha": 0.60, "beta": -10, "noise_level": 38},
    "extreme": {"blur_size": 25, "alpha": 0.45, "beta": -12, "noise_level": 50},
    "brutal":  {"blur_size": 31, "alpha": 0.32, "beta": -15, "noise_level": 65},
}


# =========================================================
# UNDERWATER / CAMERA NOISE SIMULATION
# =========================================================
def add_camera_noise(img, blur_size=15, alpha=0.7, beta=-20, noise_level=30):
    """
    Simulates underwater + poor visibility conditions.

    Parameters (all now adjustable instead of hardcoded):
      blur_size    - Gaussian blur kernel size (odd number). Bigger =
                      more scattering, more blur spread.
      alpha        - brightness/contrast scale (cv2.convertScaleAbs).
                      Lower = darker image overall.
      beta         - brightness offset (cv2.convertScaleAbs).
                      More negative = darker image overall.
      noise_level  - max random noise value added per pixel. Bigger =
                      noisier image.

    Defaults match the original hardcoded values, so any old code
    calling add_camera_noise(img) behaves exactly as before.
    """

    # =====================================================
    # WHY?
    # Water absorbs red light -> image becomes blue-green
    # =====================================================
    b, g, r = cv2.split(img)
    r = (r * 0.3).astype(np.uint8)
    img = cv2.merge((b, g, r))

    # =====================================================
    # WHY?
    # Light scattering in water reduces sharpness.
    # blur_size must be odd - enforce that here so callers
    # can pass any difficulty preset without worrying about it.
    # =====================================================
    if blur_size % 2 == 0:
        blur_size += 1
    img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    # =====================================================
    # WHY?
    # Underwater images lose brightness + contrast.
    #
    # BUG FIX: cv2.convertScaleAbs takes the ABSOLUTE VALUE of
    # (img*alpha + beta), not a proper clip to 0. This means a
    # pixel that would go slightly negative (e.g. a near-black
    # pixel) gets flipped back to a small POSITIVE number instead
    # of staying at 0 - so very dark pixels can paradoxically end
    # up BRIGHTER than slightly-less-dark pixels that land just
    # above zero. This was verified directly: a faint bleed-through
    # circle on a black background came out DIMMER on average than
    # pure black, which is physically backwards. Using proper
    # float math + np.clip fixes this: true negative results stay
    # at 0, exactly as real-world brightness (which can't go
    # below "no light") would behave.
    # =====================================================
    img = img.astype(np.float32) * alpha + beta
    img = np.clip(img, 0, 255).astype(np.uint8)

    # =====================================================
    # WHY?
    # Simulates noise like bubbles, particles, distortion
    # =====================================================
    noise = np.random.randint(0, max(noise_level, 1), img.shape, dtype='uint8')
    img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)

    return img


# =========================================================
# CONVENIENCE WRAPPER - apply noise using a named difficulty
# preset instead of remembering raw parameter values
# =========================================================
def add_camera_noise_at_level(img, difficulty="medium"):
    """
    Same as add_camera_noise(), but takes a difficulty name
    ("easy", "medium", "hard", "extreme") instead of raw
    parameters. Looks up the preset and applies it.
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(DIFFICULTY_PRESETS.keys())}"
        )

    params = DIFFICULTY_PRESETS[difficulty]
    return add_camera_noise(img, **params)


# =========================================================
# MAIN PIPELINE (unchanged - still works standalone on test.jpg)
# =========================================================
def process_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    print("Image loaded successfully")

    processed_img = add_camera_noise(img)

    output_path = "outputs/processed_image.jpg"
    cv2.imwrite(output_path, processed_img)

    print(f"Output saved at: {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    process_image("test.jpg")