import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 64
SEQUENCE_LENGTH = 20

# =====================================================
# WHY THIS ARCHITECTURE?
# TimeDistributed(CNN) applies the SAME CNN to every frame in the
# sequence independently (shared weights - it's the same kind of
# image each time, so no need for separate CNNs per position).
# This produces one feature vector per frame. Those feature
# vectors, IN ORDER, then feed into an LSTM, which can use
# information from earlier frames when interpreting later ones -
# this is what lets the model handle inter-symbol interference,
# unlike a plain per-frame CNN which has no memory at all.
# =====================================================

def build_cnn_lstm_model():
    model = models.Sequential([

        # ---- CNN applied to each frame independently (shared weights) ----
        layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation='relu'),
            input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
        ),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Flatten()),
        layers.TimeDistributed(layers.Dense(64, activation='relu')),
        # at this point: one 64-number feature vector PER FRAME,
        # shape = (batch, SEQUENCE_LENGTH, 64)

        # ---- LSTM processes the sequence of feature vectors ----
        # return_sequences=True: we want one output PER bit
        # position, not just a single summary of the whole sequence
        layers.LSTM(64, return_sequences=True),

        # ---- final per-frame bit decision ----
        layers.TimeDistributed(layers.Dense(1, activation='sigmoid')),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =====================================================
# LOAD A SEQUENCE DATASET FROM DISK
#
# WHY LOAD EVERYTHING INTO MEMORY?
# 2000 sequences x 20 frames x 64x64x3 is a manageable size
# (a few hundred MB) - simpler than writing a custom generator,
# and fast enough for this dataset size.
# =====================================================

def load_sequence_dataset(folder):
    sequence_folders = sorted(os.listdir(folder))

    X = []  # frames
    y = []  # labels

    for seq_name in sequence_folders:
        seq_path = os.path.join(folder, seq_name)

        label_path = os.path.join(seq_path, "labels.txt")
        if not os.path.exists(label_path):
            continue  # skip anything that isn't a real sequence folder

        with open(label_path) as f:
            bitstring = f.read().strip()

        frames = []
        for i in range(len(bitstring)):
            frame_path = os.path.join(seq_path, f"frame_{i:02d}.jpg")
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # BGR -> RGB fix, same as cnn_decoder_pipeline.py -
            # critical, since this is exactly the bug that broke
            # the single-frame CNN before
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img.astype("float32") / 255.0)

        X.append(frames)
        y.append([int(b) for b in bitstring])

    X = np.array(X)  # shape: (num_sequences, SEQUENCE_LENGTH, 64, 64, 3)
    y = np.array(y)  # shape: (num_sequences, SEQUENCE_LENGTH)
    y = np.expand_dims(y, axis=-1)  # shape: (num_sequences, SEQUENCE_LENGTH, 1)

    return X, y


# =====================================================
# TRAIN
# =====================================================

def train_cnn_lstm(difficulty="medium"):
    print(f"Loading sequence dataset for difficulty: {difficulty}...")

    X_train, y_train = load_sequence_dataset(f"dataset_seq/{difficulty}/train")
    X_test, y_test = load_sequence_dataset(f"dataset_seq/{difficulty}/test")

    print(f"Train sequences: {X_train.shape[0]}, Test sequences: {X_test.shape[0]}")
    print(f"Input shape per batch: {X_train.shape[1:]}")

    model = build_cnn_lstm_model()
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=16,
        callbacks=[early_stop]
    )

    model_path = f"cnn_lstm_{difficulty}.h5"
    model.save(model_path)
    print(f"Saved model: {model_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    import sys

    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    train_cnn_lstm(difficulty)