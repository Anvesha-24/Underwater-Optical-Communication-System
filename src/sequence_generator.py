import cv2
import numpy as np
from camera_noise import add_camera_noise_at_level, DIFFICULTY_PRESETS

IMG_SIZE = 64

# =====================================================
# WHY THIS FILE?
# Generates whole SEQUENCES of frames (one sequence = one
# message's worth of bits) with two realistic degradation
# effects applied:
#
# 1. INTER-SYMBOL INTERFERENCE (ISI / bleed-through):
#    Light from one bit's flash bleeds into the next frame
#    because scattered light takes time to dissipate in water.
#
# 2. MOTION (new):
#    In real underwater scenarios, neither phone is perfectly
#    still - the flashlight blob drifts and jitters across the
#    camera's field of view over time. This models that effect
#    with two components:
#      - smooth drift: the blob moves continuously in one
#        direction (like slow relative movement between phones)
#      - random jitter: small per-frame random position shifts
#        (like camera shake or turbulence)
#    motion_strength=0.0 means no motion; higher values mean
#    more drift and jitter per frame.
# =====================================================

BLEED_STRENGTH_BY_DIFFICULTY = {
    "easy":    0.05,
    "medium":  0.15,
    "hard":    0.30,
    "extreme": 0.45,
    "brutal":  0.60,
}

# pixels of drift per frame and jitter range, per motion_strength unit
# e.g. motion_strength=1.0 -> ~1.5px drift/frame + up to 2px jitter
DRIFT_SCALE  = 1.5   # pixels per frame per unit of motion_strength
JITTER_SCALE = 2.0   # max random jitter (px) per unit of motion_strength


def draw_circle_layer(x, y, radius, brightness):
    """Draws just the bright circle on its own blank canvas,
    so we can reuse this layer for bleed-through into the next
    frame, separately from compositing it into the current one."""
    layer = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cv2.circle(layer, (x, y), radius, (brightness,) * 3, -1)
    return layer


def generate_sequence(bitstring, difficulty="medium", motion_strength=0.0):
    """
    Generates a list of noisy frames for a full bitstring.

    Parameters:
      bitstring      - the bits to transmit, e.g. "10110..."
      difficulty     - noise/scattering preset name
      motion_strength - 0.0 = no motion; 1.0+ = increasing drift
                        and jitter of the flashlight blob position

    Returns: list of frames (numpy arrays), same length as bitstring.
    """
    bleed_strength = BLEED_STRENGTH_BY_DIFFICULTY[difficulty]

    # =====================================================
    # MOTION SETUP
    # Pick a random initial blob position and a random drift
    # direction. The blob will drift in this direction every
    # frame, plus random jitter, clamped to stay within bounds.
    # WHY random direction? Real motion can be in any direction,
    # not just horizontal or vertical, so we pick a random angle.
    # =====================================================
    blob_x = float(np.random.randint(20, 44))
    blob_y = float(np.random.randint(20, 44))
    radius  = np.random.randint(8, 13)
    brightness = np.random.randint(220, 256)

    # random drift direction (unit vector)
    drift_angle = np.random.uniform(0, 2 * np.pi)
    drift_dx = np.cos(drift_angle) * DRIFT_SCALE * motion_strength
    drift_dy = np.sin(drift_angle) * DRIFT_SCALE * motion_strength

    frames = []
    previous_circle_layer = None

    for bit in bitstring:
        base = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # =====================================================
        # APPLY MOTION: update blob position each frame
        # drift moves consistently in one direction; jitter adds
        # small random perturbations on top of that drift.
        # We clamp to [radius+1, IMG_SIZE-radius-1] so the blob
        # never goes off the edge of the image.
        # =====================================================
        if motion_strength > 0.0:
            jitter_x = np.random.uniform(-JITTER_SCALE, JITTER_SCALE) * motion_strength
            jitter_y = np.random.uniform(-JITTER_SCALE, JITTER_SCALE) * motion_strength
            blob_x += drift_dx + jitter_x
            blob_y += drift_dy + jitter_y

            # clamp to stay within image bounds
            margin = radius + 2
            blob_x = float(np.clip(blob_x, margin, IMG_SIZE - margin))
            blob_y = float(np.clip(blob_y, margin, IMG_SIZE - margin))

            # if we hit a boundary, reverse that drift component
            # so the blob bounces rather than getting stuck at the edge
            if blob_x <= margin or blob_x >= IMG_SIZE - margin:
                drift_dx *= -1
            if blob_y <= margin or blob_y >= IMG_SIZE - margin:
                drift_dy *= -1

        cx = int(round(blob_x))
        cy = int(round(blob_y))

        # add bleed from previous frame first
        if previous_circle_layer is not None:
            bleed = (previous_circle_layer.astype(np.float32) * bleed_strength).astype(np.uint8)
            base = cv2.add(base, bleed)

        current_circle_layer = None

        if bit == '1':
            current_circle_layer = draw_circle_layer(cx, cy, radius, brightness)
            base = cv2.add(base, current_circle_layer)

        noisy_frame = add_camera_noise_at_level(base, difficulty)
        frames.append(noisy_frame)
        previous_circle_layer = current_circle_layer

    return frames


# =====================================================
# Quick standalone test - prints blob position and
# brightness per frame to verify motion is working
# =====================================================
if __name__ == "__main__":
    test_bits = "10110100"

    print("=== No motion (motion_strength=0.0) ===")
    frames = generate_sequence(test_bits, "medium", motion_strength=0.0)
    for i, (bit, frame) in enumerate(zip(test_bits, frames)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"  bit[{i}]={bit}  max_brightness={np.max(gray)}")

    print("\n=== With motion (motion_strength=1.0) ===")
    frames = generate_sequence(test_bits, "medium", motion_strength=1.0)
    for i, (bit, frame) in enumerate(zip(test_bits, frames)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"  bit[{i}]={bit}  max_brightness={np.max(gray)}")

    print("\n=== Severe motion (motion_strength=3.0) ===")
    frames = generate_sequence(test_bits, "medium", motion_strength=3.0)
    for i, (bit, frame) in enumerate(zip(test_bits, frames)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"  bit[{i}]={bit}  max_brightness={np.max(gray)}")