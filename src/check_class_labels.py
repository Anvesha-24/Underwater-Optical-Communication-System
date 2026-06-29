"""
check_class_labels.py
 
WHY THIS FILE?
debug_signals.py showed the CNN outputting near-0 probability even for
frames where the flashlight is clearly ON (bright circle present).
That's a strong sign the class labels got flipped: e.g. "on"=0 and
"off"=1, instead of the "on"=1 / "off"=0 we assumed when interpreting
the model's output (prob > 0.5 -> '1').
 
This script just re-creates the same ImageDataGenerator/flow_from_directory
call from train_cnn.py and prints out class_indices, which tells us
exactly which folder Keras mapped to 0 and which to 1.
 
Run this in src/, where dataset/train exists.
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_dir = "dataset/train"
 
train_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
 
print("Class indices:", train_generator.class_indices)