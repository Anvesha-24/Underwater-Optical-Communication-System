from threshold_detector import signal_to_binary
from adaptive_threshold import adaptive_signal_to_binary

signal=[110,20,105,115,10]

fixed=signal_to_binary(signal)
adaptive,threshold=adaptive_signal_to_binary(signal)

print("Fixed threshold:",fixed);
print("Adaptive threshold:",adaptive);
print("Threshold Used:",threshold);
