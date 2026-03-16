import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import os
import re

# ========== CONFIG ==========
MODEL_PATH = r"D:/backup/projectCPE/Train_CNN_Digital-Readout_Version_5.0.2.h5"
TEST_IMAGE_DIR = r'D:\backup\projectCPE\dataset\images\test_digital'

# ========== LOAD MODEL ==========
model = tf.keras.models.load_model(MODEL_PATH)

# ========== LIST TEST FILES ==========
image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg')) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, '*.png')) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpeg'))

print(f"พบรูปทดสอบ {len(image_files)} ภาพ")

# ========== RESULT STORAGE ==========
correct = 0
total = 0
wrong_list = []

# ========== TEST LOOP ==========
for img_path in image_files:

    # -------- LOAD IMAGE --------
    img = Image.open(img_path).convert('RGB')

    img_arr = np.array(img, dtype="float32")
    # img_arr = img_arr / 255.0  # เปิดถ้าตอน train มี normalize
    img_arr = np.expand_dims(img_arr, axis=0)

    # -------- PREDICT --------
    pred = model.predict(img_arr, verbose=0)
    class_idx = np.argmax(pred, axis=1)[0]
    class_name = str(class_idx) if class_idx < 10 else "NaN"

    # -------- GET GROUND TRUTH FROM FILENAME --------
    filename = os.path.basename(img_path)

    match = re.match(r"(\d)", filename)
    if match:
        gt = match.group(1)
    else:
        gt = "NaN"

    # -------- CHECK CORRECT --------
    is_correct = (class_name == gt)

    if is_correct:
        correct += 1
    else:
        wrong_list.append((filename, gt, class_name))

    total += 1

    print(f"{filename} -> GT:{gt} | Predict:{class_name} | {'✓' if is_correct else '✗'}")

# ========== SUMMARY ==========
accuracy = (correct / total) * 100 if total > 0 else 0

print("\n========== SUMMARY ==========")
print(f"จำนวนภาพที่ประเมินได้: {total}")
print(f"ทำนายถูก: {correct}")
print(f"ทำนายผิด: {total - correct}")
print(f"Accuracy: {accuracy:.2f}%")

# ========== SHOW WRONG CASES ==========
if len(wrong_list) > 0:
    print("\nตัวอย่างภาพที่ทำนายผิด:")
    for item in wrong_list[:10]:
        print(f"{item[0]} -> GT:{item[1]} Predict:{item[2]}")