"""
ber_vs_motion_experiment.py

Compares plain CNN vs CNN-LSTM under increasing motion severity,
with fixed medium-difficulty scattering and medium bleed-through.

WHY THIS EXPERIMENT?
Motion is the third named reliability problem in the project brief
(after noise/scattering and inter-symbol interference). This test
asks: as the blob drifts and jitters more severely across frames,
does the CNN-LSTM's sequence awareness help compared to the plain
frame-by-frame CNN? Or does motion actually confuse the LSTM's
temporal context?

Expected behavior: at low motion both models should perform
similarly (blob position shifts are small, within what each CNN
learned). At higher motion, the blob may drift to positions or
patterns the CNNs weren't trained on — both models may degrade,
but we want to see whether the LSTM's sequence context gives it
any robustness advantage.
"""

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sequence_generator import generate_sequence
from ber_calculation import calculate_ber

IMG_SIZE = 64
SEQUENCE_LENGTH = 20
DIFFICULTY = "medium"
BLEED_STRENGTH = 0.15   # fixed at medium-difficulty preset
NUM_TRIALS = 50

print("Loading models...")
cnn_model  = load_model(f"flashlight_cnn_{DIFFICULTY}.h5")
lstm_model = load_model(f"cnn_lstm_{DIFFICULTY}.h5")
print("Models loaded.")


def random_bitstring(length):
    return "".join(random.choice("01") for _ in range(length))


def decode_with_cnn(frames):
    # BGR -> RGB fix, same as everywhere else in the pipeline
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    batch = np.array(rgb_frames, dtype="float32") / 255.0
    preds = cnn_model.predict(batch, verbose=0)
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)


def decode_with_cnn_lstm(frames):
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    seq = np.array(rgb_frames, dtype="float32") / 255.0
    seq = np.expand_dims(seq, axis=0)
    preds = lstm_model.predict(seq, verbose=0)[0]
    return "".join('1' if p[0] > 0.5 else '0' for p in preds)


def run_motion_experiment():
    # sweep from no motion to severe motion
    motion_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"\n{'='*65}")
    print("BER Comparison: Plain CNN vs CNN-LSTM under Increasing Motion")
    print(f"Difficulty: {DIFFICULTY}, Bleed: {BLEED_STRENGTH}, Trials: {NUM_TRIALS}")
    print(f"{'='*65}")
    print(f"{'motion_strength':<17}{'CNN BER':<12}{'CNN-LSTM BER':<15}{'winner'}")
    print("-" * 55)

    cnn_results  = []
    lstm_results = []

    for motion in motion_levels:
        cnn_total  = 0.0
        lstm_total = 0.0

        for _ in range(NUM_TRIALS):
            bitstring = random_bitstring(SEQUENCE_LENGTH)
            frames = generate_sequence(bitstring, DIFFICULTY,
                                       motion_strength=motion)

            cnn_bits  = decode_with_cnn(frames)
            lstm_bits = decode_with_cnn_lstm(frames)

            _, cnn_ber  = calculate_ber(bitstring, cnn_bits)
            _, lstm_ber = calculate_ber(bitstring, lstm_bits)

            cnn_total  += cnn_ber
            lstm_total += lstm_ber

        cnn_avg  = cnn_total  / NUM_TRIALS
        lstm_avg = lstm_total / NUM_TRIALS

        cnn_results.append(cnn_avg)
        lstm_results.append(lstm_avg)

        winner = "CNN-LSTM" if lstm_avg < cnn_avg else ("CNN" if cnn_avg < lstm_avg else "tie")
        print(f"{motion:<17.1f}{cnn_avg:<12.4f}{lstm_avg:<15.4f}{winner}")

    # =====================================================
    # PLOT
    # =====================================================
    plt.figure(figsize=(8, 5))
    plt.plot(motion_levels, cnn_results,  marker='o', color='#E24B4A',
             label='Plain CNN (frame-by-frame)', linewidth=2)
    plt.plot(motion_levels, lstm_results, marker='^', color='#0F6E56',
             label='CNN-LSTM (sequence-aware)', linewidth=2, linestyle='--')

    plt.xlabel("Motion strength (0 = no motion, 3 = severe drift + jitter)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("CNN vs CNN-LSTM: BER under Increasing Motion")
    plt.ylim(0, max(max(cnn_results), max(lstm_results)) + 0.05)
    plt.xticks(motion_levels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = "outputs/ber_vs_motion.png"
    plt.savefig(output_path, dpi=200)
    print(f"\nGraph saved to: {output_path}")
    plt.show()

    return motion_levels, cnn_results, lstm_results


if __name__ == "__main__":
    run_motion_experiment()