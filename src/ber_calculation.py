#bit error rate=no of error bits/total bits

def calculate_ber(original,received):
    errors=0
    
    for i in range(len(original)):
        
        if original[i]!=received[i]:
            errors+=1;
    ber=errors/len(original)
    
    return errors,ber


original = "10110011"
received = "10100011"
errors,ber=calculate_ber(original,received)

print("Errors:",errors)
print("BER:",ber)
