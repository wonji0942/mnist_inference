'''MNIST inference code for TensorFlow 2.15.0'''

import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# 1. 모델 로드 및 아키텍처 출력
model = tf.keras.models.load_model('saved_model/')
model.summary()

# 2. 실제 테스트 수행
print("\n---- Actual test for digits ----")

label_path = "dataset_test/testlabels/t_labels.txt"
# 상위 10개 라벨만 읽어들임
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f][:10]

cnt_correct = 0
for idx, label in enumerate(labels, start=1):
    # 이미지 불러와 전처리
    img = Image.open(f'dataset_test/testimgs/{idx}.png').convert('L')
    img = img.resize((28, 28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    # 예측
    preds = model.predict(arr)                  # (1, 10) shape
    pred_label = int(np.argmax(preds, axis=1)[0])

    print(f"label = {label}  -->  predicted = {pred_label}")
    if int(label) == pred_label:
        cnt_correct += 1

# 최종 정확도
final_acc = cnt_correct / len(labels)
print(f"\nFinal test accuracy: {final_acc:.4f}")

# TensorFlow 버전 출력
print("\n**** tensorflow version ****:", tf.__version__)

# 3. 사용자 정보 DataFrame 생성 예시
data = {
    '이름': ['신원지'],
    '학번': [2312010],
    '학과': ['데이터사이언스학과']
}
df = pd.DataFrame(data)
print("\nUser info:\n", df)
