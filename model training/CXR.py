from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from google.colab import drive  # If using Google Colab
import tensorflow as tf

def load_data(data_dir, img_size=(224, 224)):
    classes = ['normal', 'viral_pneumonia', 'bacterial_pneumonia', 'covid', 'tuberculosis']
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Load image as grayscale
                img = Image.open(img_path).convert('L')  # 'L' mode for grayscale
                # Resize image to the desired size
                img = img.resize(img_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

data_dir = '/content/drive/MyDrive/xraydata'
img_size = (224, 224)
images, labels = load_data(data_dir, img_size)

# Normalize images
images = images / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

#Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Specify the directory where you want to save the model
save_dir = '/content/drive/MyDrive/CXR_Model'
model_path = os.path.join(save_dir, 'xray_classification_model.h5')

# Save the model
model.save(model_path)

model.save('xray_classification_model.keras')

import os

# Specify the directory where you want to save the model
save_dir = '/content/drive/MyDrive/CXR_Model'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'xray_classification_model.keras')

# Save the model
model.save(model_path)

#from tensorflow.keras.models import load_model
#model = load_model('xray_classification_model.h5')
from tensorflow.keras.models import load_model

# Load the model
model = load_model('xray_classification_model.keras')

import numpy as np
from tensorflow.keras.preprocessing import image

def prepare_image(img_path, img_size=(224, 224)):
    img = Image.open(img_path).convert('L')
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)  # Add grayscale channel
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

new_image_path = '/content/test_0_16.jpeg'
new_image = prepare_image(new_image_path)
prediction = model.predict(new_image)
predicted_class = np.argmax(prediction, axis=1)
class_labels = ['normal', 'viral_pneumonia', 'bacterial_pneumonia', 'covid', 'tuberculosis']
print(f'Predicted class: {class_labels[predicted_class[0]]}')