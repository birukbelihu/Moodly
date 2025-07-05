# Check Training & Validation Set

import os
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_data_path = 'face-expression-recognition-dataset/images/train'
validation_data_path = 'face-expression-recognition-dataset/images/validation'

print(f" Training Subfolders: {os.listdir(training_data_path)}")
print(f" Validation Subfolders: {os.listdir(validation_data_path)}")

# Split The Dataset Into Training & Validation Set

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64

training_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_data_generator = ImageDataGenerator(rescale=1./255)

training_generator = training_data_generator.flow_from_directory(
   training_data_path,
   color_mode="grayscale",
   target_size=IMAGE_SIZE,
   batch_size=BATCH_SIZE,
   class_mode="categorical",
   shuffle=True
)

validation_generator = validation_data_generator.flow_from_directory(
   validation_data_path,
   color_mode="grayscale",
   target_size=IMAGE_SIZE,
   batch_size=BATCH_SIZE,
   class_mode="categorical",
   shuffle=True
)

# Print The Class Indices (angry, disgust, fear, happy etc.) Which We Need Later For Inferencing

print(training_generator.class_indices)

# Create The Facial Emotion Recognition CNN (Convolutional Neural Network)

EMOTIONS_COUNT = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(EMOTIONS_COUNT, activation="softmax"))

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train The Facial Emotion Recognition CNN By 30 Epochs

EPOCHS = 30
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Print The Model Loss & Accuracy Matrix

loss, accuracy = model.evaluate(validation_generator)
print("Training Loss:", loss)
print("Training Accuracy:", accuracy)

# Save The Trained Model In The Current Directory

model.save('moodly.h5')