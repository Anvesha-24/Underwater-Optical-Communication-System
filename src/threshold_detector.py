def signal_to_binary(signal,threshold=128):
    binary=""
    for value in signal:
        if value>=threshold:
            binary+="1"
        else:
            binary+="0"
            
    return binary

signal=[240,18,221,250,10]
binary=signal_to_binary(signal)

print("Signal:",signal)
print("Binary:",binary)            