def binary_to_text(binary):
    text=""
    
    #read binary 8 bits at a time
    for i in range(0,len(binary),8):
        byte=binary[i:i+8]
        ascii_value=int(byte,2)
        character=chr(ascii_value)
        text+=character
        
    return text

binary_data="0100100001000101010011000100110001001111"
decoded_message=binary_to_text(binary_data)

print("Binary Data:",binary_data)
print("Decoded Message:",decoded_message)
    