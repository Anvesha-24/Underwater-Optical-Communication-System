import cv2
import numpy as np
import os

# =========================================================
# Create output folder
# =========================================================
os.makedirs("outputs", exist_ok=True)


def detect_flashlight(image_path):
    """
    Detect flashlight in underwater/noisy image.
    Uses adaptive thresholding based on image brightness.
    """

    # =====================================================
    # Load image
    # =====================================================
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return

    print("✅ Image loaded")

    # =====================================================
    # Convert to grayscale
    # =====================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # Brightness statistics
    # =====================================================
    max_brightness = np.max(gray)
    mean_brightness = np.mean(gray)

    print("Max brightness:", max_brightness)
    print("Mean brightness:", mean_brightness)

    # =====================================================
    # Adaptive threshold
    # WHY?
    # Fixed threshold failed because max brightness
    # was only 105.
    # =====================================================
    threshold_value = int(max_brightness * 0.8)

    print("Using threshold:", threshold_value)

    _, thresh = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Save threshold image for debugging
    cv2.imwrite("outputs/threshold_debug.jpg", thresh)

    # =====================================================
    # Remove tiny noise regions
    # =====================================================
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # =====================================================
    # Find contours
    # =====================================================
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("❌ No flashlight detected")

        cv2.imwrite(
            "outputs/flashlight_detection.jpg",
            img
        )

        return

    # =====================================================
    # Select largest bright region
    # =====================================================
    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    print("Detected contour area:", area)

    if area < 10:
        print("❌ Region too small to be flashlight")

        cv2.imwrite(
            "outputs/flashlight_detection.jpg",
            img
        )

        return

    # =====================================================
    # Bounding box
    # =====================================================
    x, y, w, h = cv2.boundingRect(largest)

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # =====================================================
    # Center point
    # =====================================================
    cx = x + w // 2
    cy = y + h // 2

    cv2.circle(
        img,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

    # =====================================================
    # Save result
    # =====================================================
    output_path = "outputs/flashlight_detection.jpg"

    cv2.imwrite(output_path, img)

    print("✅ Flashlight detected")
    print(f"📍 Center: ({cx}, {cy})")
    print(f"💾 Saved: {output_path}")


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    detect_flashlight("outputs/processed_image.jpg")