"""WHY THIS FILE?
cnn_decoder_pipeline.py gave near-random BER (~0.4-0.6) for all three
detectors, and adaptive_threshold's computed threshold was suspiciously
low (1.1). That means something upstream is producing frames that are
too dark / not matching what we expect.
 
This script generates frames the same way the pipeline does, then
prints, side by side for each bit:
  - the original bit ('0' or '1')
  - max brightness in the frame   (what threshold_detector.py uses)
  - mean brightness in the frame  (what adaptive_threshold.py uses)
  - the CNN's raw probability     (before rounding to 0/1)
 
Comparing these columns tells us whether the bug is in frame
generation/camera_noise (frames don't look different enough between
bit=0 and bit=1), or in how we're interpreting the CNN's output.
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from encoder import text_to_binary
from camera_noise import add_camera_noise

IMG_SIZE=64
model=load_model("flashlight_cnn.h5")

def bit_to_frame(bit):
    img=np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
    if bit=='1':
        x=np.random.randint(20,44)
        y=np.random.randint(20,44)
        radius=np.random.randint(5,12)
        brightness=np.random.randint(180,256)
        cv2.circle(img,(x,y),radius,(brightness,)*3,-1)
        
        #if bit=='0' ,img stays all black->simulate flashlight off
    return add_camera_noise(img)    


#pull out both max & mean brightness from a frame
#convert to grayscale first (single brightness value) per pixel,easier to threshold on

def frame_brightness(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return int(np.max(gray)),float(np.mean(gray))

#take a short msg,convert to bits and print the diagnostic table
message = "HI"
binary = text_to_binary(message)
 
print(f"Message: {message}  ->  Binary: {binary}\n")
print(f"{'bit':<5}{'max_brightness':<16}{'mean_brightness':<18}{'cnn_prob':<10}")
print("-" * 50)

for bit in binary:
    frame = bit_to_frame(bit)
 
    max_val, mean_val = frame_brightness(frame)
 
    # CNN expects a batch of images normalized to 0-1
    # (matches ImageDataGenerator(rescale=1./255) used in train_cnn.py)
    cnn_input = np.expand_dims(frame.astype("float32") / 255.0, axis=0)
    prob = model.predict(cnn_input, verbose=0)[0][0]
 
    print(f"{bit:<5}{max_val:<16}{mean_val:<18.2f}{prob:<10.4f}")