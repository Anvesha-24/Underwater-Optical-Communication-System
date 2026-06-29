from encoder import text_to_binary
from decoder import binary_to_text
from signal_generator import binary_to_signal
from channel_noise import add_noise
from threshold_detector import signal_to_binary

message="HELLO"

#sender
binary=text_to_binary(message)

signal=binary_to_signal(binary)

#channel
noisy_signal=add_noise(signal)

#receiver
received_binary=signal_to_binary(noisy_signal)
decoded_message=binary_to_text(received_binary)

print("Original message:",message)
print("decoded message:",decoded_message)
