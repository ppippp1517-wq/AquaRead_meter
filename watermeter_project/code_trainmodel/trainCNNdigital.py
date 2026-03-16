import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os

# =========================
# CONFIG
# =========================
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

INPUT_DIR = r"D:\backup\projectCPE\neural-network-digital-counter-readout-5.0.0\ziffer_sortiert_resize"
MODEL_SAVE_PATH = r"D:\backup\projectCPE\Train_CNN_Digital-Readout_Version_5.0.2.h5"

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'NaN']
SUBDIRS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NaN"]

IMG_H = 32
IMG_W = 20
IMG_C = 3
NUM_CLASSES = 11

BATCH_SIZE = 4
EPOCHS = 80
SHIFT_RANGE = 1
BRIGHTNESS_RANGE = 0.3
ROTATION_ANGLE = 10
ZOOM_RANGE = 0.4

# =========================
# LOAD DATA
# =========================
x_data = []
y_data = []

for aktsubdir in SUBDIRS:
    files = glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.jpg')) + \
            glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.png')) + \
            glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.jpeg'))

    if aktsubdir == "NaN":
        category = 10
    else:
        category = int(aktsubdir)

    for aktfile in files:
        try:
            img = Image.open(aktfile).convert("RGB")
            img = np.array(img, dtype="float32")

            # ถ้าขนาดภาพไม่ตรง จะข้ามไฟล์นั้น
            if img.shape != (IMG_H, IMG_W, IMG_C):
                print(f"ข้ามไฟล์ {aktfile} เพราะขนาดไม่ตรง: {img.shape}")
                continue

            x_data.append(img)
            y_data.append(category)

        except Exception as e:
            print(f"Could not read file {aktfile}: {e}")

x_data = np.array(x_data, dtype="float32")
y_data = np.array(y_data)

if x_data.shape[0] == 0:
    print("ไม่พบไฟล์รูปภาพในไดเรกทอรีที่ระบุ กรุณาตรวจสอบเส้นทางของ INPUT_DIR")
    exit()

# ถ้าต้องการ normalize ให้เปิดบรรทัดนี้
# x_data = x_data / 255.0

y_data_cat = to_categorical(y_data, NUM_CLASSES)

print(f"Shape of x_data: {x_data.shape}")
print(f"Shape of y_data_cat: {y_data_cat.shape}")

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train_num, y_test_num = train_test_split(
    x_data,
    y_data,
    test_size=0.1,
    random_state=42,
    stratify=y_data
)

y_train = to_categorical(y_train_num, NUM_CLASSES)
y_test = to_categorical(y_test_num, NUM_CLASSES)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# =========================
# BUILD MODEL
# =========================
model = Sequential()
model.add(BatchNormalization(input_shape=(IMG_H, IMG_W, IMG_C)))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))

model.summary()

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
    metrics=["accuracy"]
)

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    width_shift_range=[-SHIFT_RANGE, SHIFT_RANGE],
    height_shift_range=[-SHIFT_RANGE, SHIFT_RANGE],
    brightness_range=[1 - BRIGHTNESS_RANGE, 1 + BRIGHTNESS_RANGE],
    zoom_range=[1 - ZOOM_RANGE, 1 + ZOOM_RANGE],
    rotation_range=ROTATION_ANGLE
)

train_iterator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_iterator,
    validation_data=(X_test, y_test),
    epochs=EPOCHS
)

# =========================
# PLOT LOSS
# =========================
plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# =========================
# PLOT ACCURACY
# =========================
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# =========================
# EVALUATE ON X_test
# =========================
y_pred_prob = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred_prob, axis=1)
y_true_labels = y_test_num

cm = confusion_matrix(y_true_labels, y_pred_labels)

print("\n=== Classification Report ===")
print(classification_report(y_true_labels, y_pred_labels, target_names=CLASS_NAMES, digits=4))

print("\n=== Confusion Matrix ===")
print(cm)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\n=== Test Evaluation ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.show()

# =========================
# ERROR ANALYSIS ON X_test
# =========================
mis_idx = np.where(y_true_labels != y_pred_labels)[0]
print(f"\nจำนวนภาพที่ทำนายผิดใน X_test: {len(mis_idx)}")

for i in mis_idx:
    true_class = CLASS_NAMES[y_true_labels[i]]
    pred_class = CLASS_NAMES[y_pred_labels[i]]
    conf = y_pred_prob[i][y_pred_labels[i]]
    print(f"Index {i}: True={true_class}, Pred={pred_class}, Confidence={conf:.4f}")

print("\n=== Error Pair Analysis ===")
for true_label in range(len(CLASS_NAMES)):
    for pred_label in range(len(CLASS_NAMES)):
        if true_label != pred_label and cm[true_label, pred_label] > 0:
            print(
                f"จริงเป็น {CLASS_NAMES[true_label]} "
                f"แต่ทำนายเป็น {CLASS_NAMES[pred_label]} "
                f": {cm[true_label, pred_label]} ภาพ"
            )

# =========================
# RESULT GRAPH ON ALL DATA
# =========================
res = []

subdir_eval = ["NaN", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for aktsubdir in subdir_eval:
    files = glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.jpg')) + \
            glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.png')) + \
            glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.jpeg'))

    if aktsubdir == "NaN":
        true_value = -1
    else:
        true_value = int(aktsubdir)

    for aktfile in files:
        try:
            test_image = Image.open(aktfile).convert("RGB")
            test_image = np.array(test_image, dtype="float32")

            if test_image.shape != (IMG_H, IMG_W, IMG_C):
                print(f"ข้ามไฟล์ {aktfile} เพราะขนาดไม่ตรง: {test_image.shape}")
                continue

            # ถ้าตอน train normalize ให้เปิดบรรทัดนี้ด้วย
            # test_image = test_image / 255.0

            img = np.reshape(test_image, [1, IMG_H, IMG_W, IMG_C])

            prediction = model.predict(img, verbose=0)
            pred_class = np.argmax(prediction, axis=1)[0]

            if pred_class == 10:
                pred_value = -1
            else:
                pred_value = pred_class

            diff = pred_value - true_value
            res.append(np.array([true_value, pred_value, diff]))

        except Exception as e:
            print(f"อ่านไฟล์ไม่ได้ {aktfile}: {e}")

res = np.asarray(res)

if len(res) > 0:
    plt.figure()
    plt.plot(res[:, 0])
    plt.plot(res[:, 1])
    plt.title('Result')
    plt.ylabel('Digital Value')
    plt.xlabel('#Picture')
    plt.legend(['Real', 'Model'], loc='upper left')
    plt.show()

# =========================
# SHOW MISCLASSIFIED FILES FROM ALL DATA
# =========================
print("\n=== Misclassified Files From Dataset Folder ===")
only_deviation = True

for aktsubdir in SUBDIRS:
    files = glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.jpg')) + \
            glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.png')) + \
            glob.glob(os.path.join(INPUT_DIR, aktsubdir, '*.jpeg'))

    expected_class = aktsubdir

    for aktfile in files:
        try:
            test_image = Image.open(aktfile).convert("RGB")
            test_image = np.array(test_image, dtype="float32")

            if test_image.shape != (IMG_H, IMG_W, IMG_C):
                print(f"ข้ามไฟล์ {aktfile} เพราะขนาดไม่ตรง: {test_image.shape}")
                continue

            # ถ้าตอน train normalize ให้เปิดบรรทัดนี้ด้วย
            # test_image = test_image / 255.0

            img = np.reshape(test_image, [1, IMG_H, IMG_W, IMG_C])

            prediction = model.predict(img, verbose=0)
            pred_class_idx = np.argmax(prediction, axis=1)[0]

            if pred_class_idx == 10:
                pred_class_name = "NaN"
            else:
                pred_class_name = str(pred_class_idx)

            if only_deviation:
                if pred_class_name != expected_class:
                    print(f"{aktfile} | expected: {expected_class} | predicted: {pred_class_name}")
            else:
                print(f"{aktfile} | expected: {expected_class} | predicted: {pred_class_name}")

        except Exception as e:
            print(f"อ่านไฟล์ไม่ได้ {aktfile}: {e}")