import numpy as np
import cv2
import matplotlib.pyplot as plt

def binary_to_frame(binary):
    frame=[]
    for bit in binary:
        if(bit=='1'):
            frame.append(255) #flash ON
        else:
            frame.append(0) #flash OFF
    return frame

binary="1011010"

frame=binary_to_frame(binary)
print("Binary:",binary)
print("Frame:",frame)   

#convert frame values into image
block_width=100
height=200

img=np.zeros(
    (height,len(frame)*block_width),
    dtype=np.uint8
)     

for i,value in enumerate(frame):
    img[:,i*block_width:(i+1)*block_width]=value
    
cv2.imwrite("test.jpg",img)

print("test.jpg generated")
    