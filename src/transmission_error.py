import random

def introduce_errors(binary, error_probability=0.1):

    noisy_binary = ""

    for bit in binary:

        if random.random() < error_probability:

            # Flip bit
            if bit == '0':
                noisy_binary += '1'
            else:
                noisy_binary += '0'

        else:
            noisy_binary += bit

    return noisy_binary


original = "0100100001000101"

received = introduce_errors(original)

print("Original :", original)
print("Received :", received)
            
            