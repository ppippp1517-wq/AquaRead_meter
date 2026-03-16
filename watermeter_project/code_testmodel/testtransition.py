import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import os
import re

# ================= CONFIG =================
PAIR_MODEL_PATH = r"D:/backup/projectCPE/pair_ab_keras.h5"
TEST_IMAGE_DIR = r"D:/backup/projectCPE/transition/testtransition"
NORMALIZE = True

# ================= LOAD MODEL =================
pair_model = tf.keras.models.load_model(PAIR_MODEL_PATH, compile=False)

print("โหลด Pair Model สำเร็จ")
print("input_shape =", pair_model.input_shape)
print("output_shape =", pair_model.output_shape)

_, H, W, C = pair_model.input_shape

# ================= IMAGE LIST =================
image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png")) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg")) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpeg"))

image_files = sorted(image_files)

print(f"\nพบภาพทั้งหมด {len(image_files)} ภาพ\n")

# ================= PREPROCESS =================
def preprocess(img_path):
    img = Image.open(img_path).convert("L").resize((W, H))
    arr = np.array(img, dtype=np.float32)

    if NORMALIZE:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ================= TEST =================
correct = 0
total = 0
errors = []

for path in image_files:
    fname = os.path.basename(path)

    x = preprocess(path)
    pred = pair_model.predict(x, verbose=0)[0]

    pred_class = int(np.argmax(pred))
    conf = float(np.max(pred))

    # อ่าน GT จากชื่อไฟล์แบบ pair เช่น 7_8_1.png -> GT = 7
    gt = None
    m = re.match(r"(\d)_(\d)_\d+", os.path.splitext(fname)[0])
    if m:
        gt = m.group(1)

    mark = "-"
    if gt is not None:
        total += 1
        if str(pred_class) == gt:
            correct += 1
            mark = "✓"
        else:
            errors.append((fname, gt, pred_class))
            mark = "✗"

    print(f"{fname} -> GT:{gt if gt is not None else 'N/A'} | Pred:{pred_class} | conf:{conf:.3f} | {mark}")

# ================= SUMMARY =================
if total > 0:
    acc = correct / total * 100
    print("\n========== SUMMARY ==========")
    print(f"จำนวนภาพที่ประเมินได้: {total}")
    print(f"ทำนายถูก: {correct}")
    print(f"ทำนายผิด: {total - correct}")
    print(f"Accuracy: {acc:.2f}%")

if errors:
    print("\nตัวอย่างภาพที่ทำนายผิด:")
    for e in errors[:10]:
        print(e)