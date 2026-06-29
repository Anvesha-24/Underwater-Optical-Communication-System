def text_to_binary(text):
    binary=""
    
    for char in text:
        ascii_value=ord(char)
        binary_char=format(ascii_value,'08b') #ASCII->8 bit binary
        binary+=binary_char
    return binary

message="HELLO"
binary_message=text_to_binary(message)
print("Original Message:",message)
print("Binary Message:",binary_message)