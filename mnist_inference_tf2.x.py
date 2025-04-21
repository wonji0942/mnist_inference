from __future__ import print_function
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

#─── SavedModel을 TensorFlow Core API 로드 ─────────────────────
loaded = tf.saved_model.load("saved_model/")
infer  = loaded.signatures["serving_default"]

# 입력 시그니처 이름 자동 추출
input_name = list(infer.structured_input_signature[1].keys())[0]

print("\n----Actual test for digits----\n")

#─── 라벨 파일 불러오기 ────────────────────────────────────
with open("dataset_test/testlabels/t_labels.txt","r") as f:
    labels = [line.strip() for line in f]

cnt_correct = 0

#─── 10개 이미지 순회하며 추론 ─────────────────────────────
for i, label in enumerate(labels[:10]):
    img = Image.open(f"dataset_test/testimgs/{i+1}.png").convert("L")
    img = img.resize((28,28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1,28,28,1)

    # 서명(signature) 호출 (키워드 인자로 전달)
    tf_input = tf.constant(arr)
    outputs  = infer(**{input_name: tf_input})
    y_pred   = list(outputs.values())[0].numpy()
    pred     = np.argmax(y_pred, axis=1)[0]

    print(f"label = {label} --> predicted label = {pred}")
    if int(label) == pred:
        cnt_correct += 1

#─── 정확도 및 학번/이름 출력 ──────────────────────────────
acc = cnt_correct / 10
print(f"\nFinal test accuracy: {acc:.2f}\n")
print("****tensorflow version****:", tf.__version__)

df = pd.DataFrame({
    "이름": ["신원지"],
    "학번": [2312010],
    "학과": ["데이터사이언스학과"]
})
print(df)
