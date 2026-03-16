import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import os
import math
import csv
import matplotlib.pyplot as plt

# ========== CONFIG ==========
CNN_MODEL_PATH = r"D:/backup/projectCPE/watermeter_project/meter_reader/modelneedle.h5"
TEST_IMAGE_DIR = r"D:/backup/projectCPE/watermeter_project/meter_reader/testneedle"
OUTPUT_CSV = r"D:/backup/projectCPE/watermeter_project/pointer_test_results_fixed.csv"

GT_PRED_PLOT_PATH = r"D:/backup/projectCPE/watermeter_project/gt_vs_pred_pointer.png"
ERROR_HIST_PATH = r"D:/backup/projectCPE/watermeter_project/error_hist_pointer.png"

IMG_SIZE = (32, 32)
NORMALIZE = False   # ถ้าตอน train หาร 255.0 ให้เปลี่ยนเป็น True

# ========== LOAD MODEL ==========
model = tf.keras.models.load_model(CNN_MODEL_PATH)
print("โหลดโมเดลสำเร็จ")
print("model.input_shape =", model.input_shape)
print("model.output_shape =", model.output_shape)

# ========== LIST TEST FILES ==========
image_files = glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg")) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png")) \
            + glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpeg"))

image_files = sorted(image_files)

print(f"พบรูปทดสอบ {len(image_files)} ภาพ\n")

# ========== HELPER FUNCTIONS ==========
def parse_dial_value_from_filename(filename):
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    value_str = name_no_ext.split("_")[0]
    return float(value_str)

def dial_value_to_angle(value):
    return (value / 10.0) * 360.0

def predict_angle_from_output(pred):
    pred = np.array(pred).flatten()

    if len(pred) != 2:
        raise ValueError(f"คาดว่า output ต้องมี 2 ค่า แต่ได้ shape = {pred.shape}")

    sin_theta = float(pred[0])
    cos_theta = float(pred[1])

    angle_deg = math.degrees(math.atan2(sin_theta, cos_theta))
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg, sin_theta, cos_theta

def angular_error_deg(gt, pred):
    diff = abs(gt - pred) % 360
    return min(diff, 360 - diff)

# ========== TEST LOOP ==========
results = []
errors = []
gt_angles = []
pred_angles = []
filenames = []

for img_path in image_files:
    try:
        dial_value = parse_dial_value_from_filename(img_path)
        gt_angle = dial_value_to_angle(dial_value)

        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)

        img_arr = np.array(img, dtype="float32")
        if NORMALIZE:
            img_arr = img_arr / 255.0

        img_arr = np.expand_dims(img_arr, axis=0)

        pred = model.predict(img_arr, verbose=0)[0]
        pred_angle, pred_sin, pred_cos = predict_angle_from_output(pred)

        err = angular_error_deg(gt_angle, pred_angle)

        errors.append(err)
        gt_angles.append(gt_angle)
        pred_angles.append(pred_angle)
        filenames.append(os.path.basename(img_path))

        print(
            f"{os.path.basename(img_path)} -> "
            f"Value: {dial_value:.2f} | GT angle: {gt_angle:.2f}° | "
            f"Pred: {pred_angle:.2f}° | Error: {err:.2f}°"
        )

        results.append([
            os.path.basename(img_path),
            round(dial_value, 2),
            round(gt_angle, 2),
            round(pred_sin, 6),
            round(pred_cos, 6),
            round(pred_angle, 2),
            round(err, 2)
        ])

    except Exception as e:
        print(f"{os.path.basename(img_path)} -> ERROR: {e}")

# ========== SUMMARY ==========
if len(errors) > 0:
    errors = np.array(errors)

    mae = np.mean(errors)
    acc_3 = np.mean(errors <= 3) * 100
    acc_5 = np.mean(errors <= 5) * 100
    acc_10 = np.mean(errors <= 10) * 100

    print("\n========== SUMMARY ==========")
    print(f"จำนวนภาพที่ประเมินได้: {len(errors)}")
    print(f"MAE: {mae:.2f}°")
    print(f"Accuracy within ±3° : {acc_3:.2f}%")
    print(f"Accuracy within ±5° : {acc_5:.2f}%")
    print(f"Accuracy within ±10°: {acc_10:.2f}%")

# ========== SAVE CSV ==========
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "dial_value",
        "ground_truth_angle_deg",
        "pred_sin",
        "pred_cos",
        "pred_angle_deg",
        "abs_error_deg"
    ])
    writer.writerows(results)

print(f"\nบันทึกผลลัพธ์ลงไฟล์: {OUTPUT_CSV}")

# ========== PLOT 1: GT vs Prediction ==========
if len(gt_angles) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(gt_angles, label="Ground Truth Angle")
    plt.plot(pred_angles, label="Predicted Angle")
    plt.xlabel("Test Image Index")
    plt.ylabel("Angle (degrees)")
    plt.title("Ground Truth Angle vs Predicted Angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GT_PRED_PLOT_PATH, dpi=300)
    plt.show()

    print(f"บันทึกกราฟ GT vs Pred ไฟล์: {GT_PRED_PLOT_PATH}")

# ========== PLOT 2: Error Histogram ==========
if len(errors) > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=15, edgecolor="black")
    plt.xlabel("Absolute Angular Error (degrees)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Angular Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ERROR_HIST_PATH, dpi=300)
    plt.show()

    print(f"บันทึกกราฟ Error Histogram ไฟล์: {ERROR_HIST_PATH}")