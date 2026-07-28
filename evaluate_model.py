import os

from PIL import Image
from tf_keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from util import classify

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = './train'  # expects ./train/<class_name>/*.jpg, matching labels.txt

model = load_model('./model/pneumonia_classifier.h5')

with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f if line.strip()]

print('Classes (in model order):', class_names)

# ---------------------------------------------------------------------------
# Run the model over every image and compare its prediction to the folder
# it came from
# ---------------------------------------------------------------------------
y_true = []
y_pred = []
mistakes = []

for true_index, class_name in enumerate(class_names):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f'Warning: no folder found for class "{class_name}" at {class_dir}')
        continue

    files = os.listdir(class_dir)
    print(f'\nEvaluating {len(files)} images in "{class_name}"...')

    for i, file in enumerate(files):
        img_path = os.path.join(class_dir, file)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        predicted_name, confidence = classify(image, model, class_names)
        predicted_index = class_names.index(predicted_name)

        y_true.append(true_index)
        y_pred.append(predicted_index)

        if predicted_index != true_index:
            mistakes.append((img_path, class_name, predicted_name, confidence))

        if (i + 1) % 100 == 0:
            print(f'  ...{i + 1}/{len(files)}')

# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
print(f'\nOverall accuracy on your dataset: {accuracy * 100:.2f}%')

print('\nClassification report:')
print(classification_report(y_true, y_pred, target_names=class_names))

print('Confusion matrix (rows = true, columns = predicted):')
print(confusion_matrix(y_true, y_pred))

print(f'\n{len(mistakes)} misclassified images. First 10:')
for img_path, true_name, predicted_name, confidence in mistakes[:10]:
    print(f'  {img_path}: true={true_name}, predicted={predicted_name} ({confidence * 100:.1f}%)')