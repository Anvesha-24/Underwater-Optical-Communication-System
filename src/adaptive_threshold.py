def adaptive_signal_to_binary(signal):
    threshold=sum(signal)/len(signal)
    binary=""
    
    for value in signal:
        if value>threshold:
            binary+="1"
        else:
            binary+="0"
    return binary,threshold

signal=[110,20,105,115,10]

binary,threshold=adaptive_signal_to_binary(signal)

print("Signal:",signal)
print("Adaptive Threshold:",threshold)
print("Binary:",binary)
            