#0   = completely dark
#255 = maximum brightness
def binary_to_signal(binary):
    signal=[]
    for bit in binary:
        if bit=='1':
            signal.append(255) #flash ON
            
        else:
            signal.append(0) #flash OFF
            
    return signal

binary = "10110010"

signal=binary_to_signal(binary)       

print("Binary:",binary)
print("Signal:",signal)
