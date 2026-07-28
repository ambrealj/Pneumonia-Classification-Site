import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)


def classify(image, model, class_names):
    """
    Resize to 224x224, normalize to [-1, 1] (the range MobileNetV2 / the
    Teachable Machine export expects), and run the model. Supports both
    single-output binary models and multi-output softmax models.
    """
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1,) + IMG_SIZE + (3,), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = np.asarray(model.predict(data, verbose=0))

    if prediction.ndim == 2 and prediction.shape[1] > 1:
        index = int(np.argmax(prediction[0]))
        confidence_score = float(prediction[0][index])
    else:
        probability = float(prediction[0][0])
        index = 0 if probability > 0.5 else 1
        confidence_score = probability if index == 0 else 1.0 - probability

    return class_names[index], confidence_score