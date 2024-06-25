from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# Check if model path exists
model_path = 'model1'
print(os.path.exists(model_path))

# Load the TensorFlow model
model = tf.saved_model.load(model_path)

# List of disease classes
disease_class = ['NORMAL', 'VIRAL PNEUMONIA', 'BACTERIAL PNEUMONIA', 'COVID', 'TUBERCULOSIS']

def prepare_image(img, img_size=(224, 224)):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize(img_size)  # Resize image
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)  # Add grayscale channel
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.astype(np.float32)  # Convert to float32
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Load the image from the request
    image_file = request.files['image']
    img = Image.open(io.BytesIO(image_file.read()))  # Pass file content to Image.open
    
    # Image preprocessing and prediction
    prepared_image = prepare_image(img)
    
    # Make prediction
    predictions = model(prepared_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = disease_class[predicted_class_index]
    confidence_percentage = predictions.numpy()[0][predicted_class_index] * 100
   
    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence_percentage
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
