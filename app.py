import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
from tf_keras.models import load_model

app = Flask(__name__)

model = load_model('./model/pneumonia_classifier.h5')

with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f if line.strip()]


def classify(image, model, class_names):
    """
    Same logic as the original util.classify(): resize to 224x224, normalize
    to [-1, 1], run the model, and support both single-output binary models
    and multi-output softmax models.
    """
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = np.asarray(model.predict(data))

    if prediction.ndim == 2 and prediction.shape[1] > 1:
        index = int(np.argmax(prediction[0]))
        confidence_score = float(prediction[0][index])
    else:
        probability = float(prediction[0][0])
        index = 0 if probability > 0.5 else 1
        confidence_score = probability if index == 0 else 1.0 - probability

    return class_names[index], confidence_score


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read image file'}), 400

    class_name, confidence = classify(image, model, class_names)

    return jsonify({
        'class_name': class_name,
        'confidence': round(confidence * 100, 1)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)