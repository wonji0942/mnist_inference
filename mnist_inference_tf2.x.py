'''This is inference code for MNIST dataset'''

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os

# Load the trained model
model = load_model('saved_model/')

# Show model architecture
model.summary()

# ---- Actual test for digits ----
print("\n---- Actual test for digits ----")

mnist_label_file_path = "dataset_test/testlabels/t_labels.txt"
with open(mnist_label_file_path, "r") as mnist_label:
    labels = mnist_label.readlines()

cnt_correct = 0

for index in range(10):
    label = labels[index].strip()

    # Load and preprocess image
    img_path = f'dataset_test/testimgs/{index+1}.png'
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img).astype('float32') / 255.0
    im2arr = im2arr.reshape(1, 28, 28, 1)

    # Predict the class
    y_pred = model.predict(im2arr)
    pred_label = np.argmax(y_pred)

    print(f"label = {label} --> predicted label = {pred_label}")

    # Compute accuracy
    if int(label) == pred_label:
        cnt_correct += 1

# Final accuracy
final_acc = cnt_correct / 10
print(f"\nFinal test accuracy: {final_acc:.2f}")
print(f"\n**** TensorFlow version: {tf.__version__} ****\n")

# Student info output
data = {
    '이름': ['신원지'],
    '학번': [2312010],
    '학과': ['데이터사이언스학과']
}
df = pd.DataFrame(data)
print(df)
