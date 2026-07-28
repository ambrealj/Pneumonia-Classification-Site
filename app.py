import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from tf_keras.models import load_model

from util import classify

app = Flask(__name__)

model = load_model('./model/pneumonia_classifier.h5')

with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f if line.strip()]


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