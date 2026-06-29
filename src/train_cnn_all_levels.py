import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import cv2

# =====================================================
# WHY ONE CNN PER DIFFICULTY LEVEL?
# A single CNN trained only on "easy" data has no reason to
# perform well on "hard"/"extreme" noise - it never saw that
# distribution. To fairly test "does ML degrade more
# gracefully than threshold methods as scattering increases",
# each difficulty level needs its own CNN trained on data
# from that same level. This mirrors a real deployment where
# you'd tune/retrain for expected water conditions.
# =====================================================

DIFFICULTIES = ["easy", "medium", "hard", "extreme"]


def build_model():
    """Same architecture as before - the issue was never the
    architecture, it was data/preprocessing, so we keep this
    unchanged and trust it across difficulty levels."""

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def check_class_accuracy(model, folder, label, num_samples=200):
    """Loads images via PIL-equivalent (RGB) since that's what
    the model trains/expects - using cv2 then converting to RGB
    to stay consistent with the channel-order fix from
    cnn_decoder_pipeline.py."""

    files = sorted(os.listdir(folder))[:num_samples]
    images = []

    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        img = cv2.resize(img, (64, 64))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb.astype("float32") / 255.0)

    batch = np.array(images)
    preds = model.predict(batch, verbose=0).flatten()

    if label == "on":
        correct = np.sum(preds > 0.5)
    else:
        correct = np.sum(preds < 0.5)

    print(f"  {label.upper()} accuracy: {correct}/{len(preds)}  (mean prob: {preds.mean():.4f})")
    return correct / len(preds)


def train_one_difficulty(difficulty):
    print(f"\n{'='*50}")
    print(f"TRAINING MODEL FOR DIFFICULTY: {difficulty}")
    print(f"{'='*50}")

    train_dir = f"dataset/{difficulty}/train"
    test_dir = f"dataset/{difficulty}/test"

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(64, 64), batch_size=32,
        class_mode='binary', shuffle=True, seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(64, 64), batch_size=32,
        class_mode='binary', shuffle=False
    )

    print("Class indices:", train_generator.class_indices)

    model = build_model()

    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=20,
        callbacks=[early_stop],
        verbose=1
    )

    model_path = f"flashlight_cnn_{difficulty}.h5"
    model.save(model_path)
    print(f"Saved model: {model_path}")

    print("\n--- Per-class accuracy on test set ---")
    on_acc = check_class_accuracy(model, f"dataset/{difficulty}/test/on", "on")
    off_acc = check_class_accuracy(model, f"dataset/{difficulty}/test/off", "off")

    return on_acc, off_acc


def train_all_difficulties():
    results = {}

    for difficulty in DIFFICULTIES:
        on_acc, off_acc = train_one_difficulty(difficulty)
        results[difficulty] = (on_acc, off_acc)

    print(f"\n{'='*50}")
    print("SUMMARY - per-class accuracy by difficulty")
    print(f"{'='*50}")
    print(f"{'difficulty':<12}{'on_acc':<10}{'off_acc':<10}")
    for difficulty, (on_acc, off_acc) in results.items():
        print(f"{difficulty:<12}{on_acc:<10.2%}{off_acc:<10.2%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # e.g. python train_cnn_all_levels.py brutal
        difficulty = sys.argv[1]
        on_acc, off_acc = train_one_difficulty(difficulty)
        print(f"\n{difficulty}: on_acc={on_acc:.2%}  off_acc={off_acc:.2%}")
    else:
        train_all_difficulties()