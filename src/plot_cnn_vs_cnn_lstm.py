"""
plot_cnn_vs_cnn_lstm.py

Plots the BER comparison between plain CNN and CNN-LSTM under
increasing inter-symbol interference (bleed strength), from the
results of ber_cnn_vs_cnn_lstm.py.

Run from src/ - saves PNG to outputs/ber_cnn_vs_cnn_lstm.png
"""

import matplotlib.pyplot as plt

# =====================================================
# RESULTS FROM ber_cnn_vs_cnn_lstm.py
# Update these if you rerun the experiment
# =====================================================

bleed_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

cnn_ber  = [0.0000, 0.0000, 0.0730, 0.2380, 0.2250, 0.2310, 0.2360]
lstm_ber = [0.0000, 0.0000, 0.0000, 0.0000, 0.0850, 0.1690, 0.2240]

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(bleed_levels, cnn_ber,  marker='o', color='#E24B4A',
         label='Plain CNN (frame-by-frame)', linewidth=2)
plt.plot(bleed_levels, lstm_ber, marker='^', color='#0F6E56',
         label='CNN-LSTM (sequence-aware)', linewidth=2, linestyle='--')

# mark the crossover point where CNN first starts failing
plt.axvline(x=0.2, color='gray', linestyle=':', linewidth=1.2,
            label='ISI crossover point (bleed=0.2)')

plt.xlabel("Bleed strength (inter-symbol interference level)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("CNN vs CNN-LSTM: BER under Increasing Inter-Symbol Interference")
plt.ylim(0, 0.35)
plt.xticks(bleed_levels)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# =====================================================
# SAVE
# =====================================================

output_path = "outputs/ber_cnn_vs_cnn_lstm.png"
plt.savefig(output_path, dpi=200)
print(f"Graph saved to: {output_path}")

plt.show()