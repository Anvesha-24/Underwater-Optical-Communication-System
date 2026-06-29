"""
plot_ber_results.py

Takes the BER-vs-difficulty results from ber_vs_difficulty_experiment.py
and plots them as a line graph, saved as a PNG - useful for including
directly in your report/slides for your mentor.

Run this in src/ (or anywhere - it doesn't depend on the dataset or
trained models, just the numbers below). Requires matplotlib:
    pip install matplotlib --break-system-packages   (if not already installed)
"""

import matplotlib.pyplot as plt

# =====================================================
# RESULTS FROM ber_vs_difficulty_experiment.py
#
# WHY HARDCODED?
# These are the actual numbers you already got from running the
# real experiment in VS Code. If you rerun the experiment later
# (e.g. after adding more difficulty levels or the CNN-LSTM model),
# just update these lists with the new printed values.
# =====================================================

difficulty_levels = ["Easy", "Medium", "Hard", "Extreme", "Brutal"]

fixed_ber    = [0.076, 0.507, 0.486, 0.499, 0.497]
adaptive_ber = [0.000, 0.000, 0.000, 0.454, 0.497]
cnn_ber      = [0.000, 0.000, 0.000, 0.000, 0.000]

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(8, 5))

# different markers per line, not just color, so the graph
# is still readable in black-and-white printouts
plt.plot(difficulty_levels, fixed_ber, marker='o', color='#E24B4A',
          label='Fixed threshold', linewidth=2)
plt.plot(difficulty_levels, adaptive_ber, marker='s', color='#BA7517',
          label='Adaptive threshold', linewidth=2, linestyle='--')
plt.plot(difficulty_levels, cnn_ber, marker='^', color='#0F6E56',
          label='CNN', linewidth=2, linestyle=':')

plt.xlabel("Difficulty level (increasing scattering / noise)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs Difficulty: Fixed Threshold vs Adaptive Threshold vs CNN")
plt.ylim(0, 0.6)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# =====================================================
# SAVE
# =====================================================

output_path = "outputs/ber_vs_difficulty.png"
plt.savefig(output_path, dpi=200)
print(f"Graph saved to: {output_path}")

plt.show()