import logging
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('model.h5')

# Initialize Flask app
app = Flask(__name__)

# Set up logging with UTF-8 encoding
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    handlers=[
                        logging.FileHandler("app.log", encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Define class names based on your dataset
class_indices = {
    0: 'Normal',
    1: 'Otitis',
    2: 'Other Disease'
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)
    
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_indices[np.argmax(predictions[0])]

    # Log the prediction with UTF-8 support
    logging.info(f"Predicted class: {predicted_class}")
    
    return f"The predicted class is: {predicted_class}"

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
