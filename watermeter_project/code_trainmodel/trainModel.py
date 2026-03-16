# -*- coding: utf-8 -*-
"""
train_compare_models.py

Compare digit classification models on water-meter digit dataset:
1) Baseline CNN
2) MobileNetV2
3) ResNet50

Dataset structure:
DATA_ROOT/
    0/
    1/
    2/
    ...
    9/
    NaN/

Outputs:
- best model for each architecture
- training history CSV
- training curves (accuracy / loss)
- confusion matrix
- classification report
- summary CSV comparing all models

Author: OpenAI ChatGPT
"""

import os
import re
import json
import time
import random
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = r"D:\backup\projectCPE\digit_datasetmodel"   # <-- แก้ path ตรงนี้
OUT_DIR   = r"D:\backup\projectCPE\reports\digit_compare"

CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NaN"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_H = 64
IMG_W = 64
CHANNELS = 3   # RGB input

TEST_RATIO = 0.10
VAL_RATIO_FROM_REMAIN = 0.1111111111   # approx 80/10/10 split
SEED = 42

EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3

USE_AUGMENT = True
LABEL_SMOOTHING = 0.0

# If True, MobileNetV2 / ResNet50 will use ImageNet weights.
# For small/specific datasets, False is often safer at first.
USE_IMAGENET_WEIGHTS = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# =========================================================
# UTILITIES
# =========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def natural_key(path: str):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return [int(x) for x in nums] if nums else [base]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resize_keep_pad(im: Image.Image, out_w: int, out_h: int, fill_color=0) -> Image.Image:
    w, h = im.size
    scale = min(out_w / max(1, w), out_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = im.resize((new_w, new_h), Image.BILINEAR)

    if im.mode == "RGB":
        canvas = Image.new("RGB", (out_w, out_h), color=(fill_color, fill_color, fill_color))
    else:
        canvas = Image.new(im.mode, (out_w, out_h), color=fill_color)

    off_x = (out_w - new_w) // 2
    off_y = (out_h - new_h) // 2
    canvas.paste(im, (off_x, off_y))
    return canvas


def load_image_rgb(path: str, out_w: int, out_h: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im = resize_keep_pad(im, out_w, out_h, fill_color=0)
    arr = np.array(im, dtype=np.float32)

    # กันพลาดอีกชั้น
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr


def simple_augment_np(img: np.ndarray) -> np.ndarray:
    """
    Light augmentation for numpy image in [0,1], shape (H,W,3).
    """
    out = img.copy()

    # brightness / contrast
    if random.random() < 0.7:
        alpha = 1.0 + (random.random() * 0.2 - 0.1)  # 0.9..1.1
        beta = random.random() * 0.10 - 0.05         # -0.05..0.05
        out = np.clip(out * alpha + beta, 0.0, 1.0)

    # small shifts
    if random.random() < 0.5:
        dy = random.randint(-2, 2)
        out = np.roll(out, dy, axis=0)

    if random.random() < 0.5:
        dx = random.randint(-2, 2)
        out = np.roll(out, dx, axis=1)

    # mild Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0.0, 0.02, out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    return out


# =========================================================
# DATA LOADING
# =========================================================

def collect_image_paths(data_root: str, class_names: List[str]) -> List[Tuple[str, int]]:
    items = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_root, class_name)

        if not os.path.isdir(class_dir):
            print(f"[WARNING] Class folder not found: {class_dir}")
            continue

        files = []
        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                files.append(os.path.join(class_dir, fname))

        files.sort(key=natural_key)

        for f in files:
            items.append((f, class_idx))

    return items


def load_dataset(data_root: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        X: shape (N,H,W,3), float32 in [0,1]
        y: shape (N,)
        paths: list[str]
    """
    items = collect_image_paths(data_root, CLASS_NAMES)

    if len(items) == 0:
        raise RuntimeError(f"No images found in DATA_ROOT: {data_root}")

    X, y, paths = [], [], []

    for path, label in items:
        arr = load_image_rgb(path, IMG_W, IMG_H)  # ควรได้ (H,W,3)
        X.append(arr)
        y.append(label)
        paths.append(path)

    X = np.stack(X, axis=0).astype(np.float32)

    # ===== บังคับแก้ shape ให้เป็น 4D =====
    if X.ndim == 3:
        # จาก (N,H,W) -> (N,H,W,1)
        X = np.expand_dims(X, axis=-1)

    if X.shape[-1] == 1:
        # จาก (N,H,W,1) -> (N,H,W,3)
        X = np.repeat(X, 3, axis=-1)

    # normalize
    X = X / 255.0
    y = np.array(y, dtype=np.int32)

    print("DEBUG X.shape after fix =", X.shape)
    print("DEBUG sample shape =", X[0].shape)

    return X, y, paths

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    paths: List[str],
    test_ratio: float = 0.10,
    val_ratio_from_remain: float = 0.1111111111,
    seed: int = 42
):
    """
    Stratified split into train / val / test.
    Approx train/val/test = 80/10/10
    """
    idx_all = np.arange(len(y))

    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=test_ratio,
        stratify=y,
        random_state=seed
    )

    y_trainval = y[idx_trainval]

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_ratio_from_remain,
        stratify=y_trainval,
        random_state=seed
    )

    def pick(indices):
        return X[indices], y[indices], [paths[i] for i in indices]

    return pick(idx_train), pick(idx_val), pick(idx_test)


# =========================================================
# TF DATASET
# =========================================================

def make_tf_dataset(
    X: np.ndarray,
    y_onehot: np.ndarray,
    batch_size: int,
    training: bool = False
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y_onehot))

    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED, reshuffle_each_iteration=True)

        if USE_AUGMENT:
            def aug_fn(img, label):
                img = tf.numpy_function(simple_augment_np, [img], Tout=tf.float32)
                img.set_shape((IMG_H, IMG_W, CHANNELS))
                return img, label

            ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================================================
# MODEL BUILDERS
# =========================================================

def build_baseline_cnn() -> tf.keras.Model:
    model = models.Sequential(name="BaselineCNN")
    model.add(layers.Input(shape=(IMG_H, IMG_W, CHANNELS)))

    model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.30))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model


def build_mobilenetv2() -> tf.keras.Model:
    weights = "imagenet" if USE_IMAGENET_WEIGHTS else None

    base = MobileNetV2(
        input_shape=(IMG_H, IMG_W, CHANNELS),
        include_top=False,
        weights=weights
    )

    # Freeze only if using pretrained weights initially
    base.trainable = True if not USE_IMAGENET_WEIGHTS else False

    inputs = layers.Input(shape=(IMG_H, IMG_W, CHANNELS))
    x = base(inputs, training=not USE_IMAGENET_WEIGHTS)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    return model


def build_resnet50() -> tf.keras.Model:
    weights = "imagenet" if USE_IMAGENET_WEIGHTS else None

    base = ResNet50(
        input_shape=(IMG_H, IMG_W, CHANNELS),
        include_top=False,
        weights=weights
    )

    base.trainable = True if not USE_IMAGENET_WEIGHTS else False

    inputs = layers.Input(shape=(IMG_H, IMG_W, CHANNELS))
    x = base(inputs, training=not USE_IMAGENET_WEIGHTS)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="ResNet50")
    return model


MODEL_BUILDERS = {
    "cnn": build_baseline_cnn,
    "mobilenetv2": build_mobilenetv2,
    "resnet50": build_resnet50,
}


# =========================================================
# TRAIN / EVAL / SAVE
# =========================================================

def compile_model(model: tf.keras.Model) -> None:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=loss_fn,
        metrics=["accuracy"]
    )


def make_callbacks(model_dir: str) -> List[tf.keras.callbacks.Callback]:
    ensure_dir(model_dir)
    ckpt_path = os.path.join(model_dir, "best_model.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def plot_history(history: tf.keras.callbacks.History, save_dir: str, model_name: str) -> None:
    ensure_dir(save_dir)
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        plt.plot(epochs, hist["val_accuracy"], label="val")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy.png"), dpi=180)
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["loss"], label="train")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="val")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=180)
    plt.close()

    pd.DataFrame(hist).to_csv(os.path.join(save_dir, "history.csv"), index=False)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, title: str) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_int: np.ndarray,
    class_names: List[str],
    save_dir: str,
    model_name: str
) -> Dict:
    ensure_dir(save_dir)

    start = time.perf_counter()
    pred_prob = model.predict(X_test, batch_size=64, verbose=0)
    elapsed = time.perf_counter() - start

    y_pred = np.argmax(pred_prob, axis=1)

    acc = accuracy_score(y_test_int, y_pred)
    f1_macro = f1_score(y_test_int, y_pred, average="macro")
    f1_weighted = f1_score(y_test_int, y_pred, average="weighted")
    avg_infer_ms = (elapsed / max(1, len(X_test))) * 1000.0

    cm = confusion_matrix(y_test_int, y_pred, labels=list(range(len(class_names))))
    np.savetxt(os.path.join(save_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(save_dir, "confusion_matrix.png"),
        title=f"{model_name} Confusion Matrix"
    )

    report_dict = classification_report(
        y_test_int,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(save_dir, "classification_report.csv")
    )

    result = {
        "model_name": model_name,
        "test_accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "avg_inference_ms_per_image": float(avg_infer_ms),
        "num_test_samples": int(len(X_test)),
        "num_params": int(model.count_params()),
    }

    with open(os.path.join(save_dir, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def train_one_model(
    model_key: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    X_test: np.ndarray,
    y_test_int: np.ndarray,
    save_root: str,
    class_weight: Dict[int, float]
) -> Dict:
    model_name = model_key.lower()
    model_dir = os.path.join(save_root, model_name)
    ensure_dir(model_dir)

    print("=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    model = MODEL_BUILDERS[model_name]()
    compile_model(model)

    with open(os.path.join(model_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    callbacks = make_callbacks(model_dir)

    start_train = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    train_time_sec = time.perf_counter() - start_train

    model.save(os.path.join(model_dir, "final_model.keras"))

    plot_history(history, model_dir, model_name)

    hist_df = pd.DataFrame(history.history)
    best_val_acc = float(hist_df["val_accuracy"].max()) if "val_accuracy" in hist_df else float("nan")
    best_val_loss = float(hist_df["val_loss"].min()) if "val_loss" in hist_df else float("nan")
    best_epoch = int(hist_df["val_accuracy"].idxmax() + 1) if "val_accuracy" in hist_df else len(hist_df)

    eval_result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test_int=y_test_int,
        class_names=CLASS_NAMES,
        save_dir=model_dir,
        model_name=model_name
    )

    eval_result["best_val_accuracy"] = best_val_acc
    eval_result["best_val_loss"] = best_val_loss
    eval_result["best_epoch"] = best_epoch
    eval_result["train_time_sec"] = float(train_time_sec)

    with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    return eval_result


def save_comparison_table(results: List[Dict], out_dir: str) -> None:
    ensure_dir(out_dir)

    df = pd.DataFrame(results)
    df = df[
        [
            "model_name",
            "best_val_accuracy",
            "best_val_loss",
            "best_epoch",
            "test_accuracy",
            "f1_macro",
            "f1_weighted",
            "avg_inference_ms_per_image",
            "num_params",
            "train_time_sec",
            "num_test_samples",
        ]
    ]
    df.sort_values(by="test_accuracy", ascending=False, inplace=True)
    df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)

    # Test accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(df["model_name"], df["test_accuracy"])
    plt.title("Model Comparison - Test Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_test_accuracy.png"), dpi=180)
    plt.close()

    # Inference time comparison
    plt.figure(figsize=(8, 5))
    plt.bar(df["model_name"], df["avg_inference_ms_per_image"])
    plt.title("Model Comparison - Avg Inference Time per Image")
    plt.xlabel("Model")
    plt.ylabel("Milliseconds / image")
    plt.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_inference_time.png"), dpi=180)
    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Loading dataset from: {DATA_ROOT}")

    X, y_int, paths = load_dataset(DATA_ROOT)

    print(f"Loaded dataset: {X.shape}, labels: {y_int.shape}")
    print("Sample image shape:", X[0].shape)
    print("Class distribution:")
    for i, cname in enumerate(CLASS_NAMES):
        cnt = int((y_int == i).sum())
        print(f"  {cname:>4s}: {cnt}")

    # Split
    (X_train, y_train_int, p_train), (X_val, y_val_int, p_val), (X_test, y_test_int, p_test) = split_dataset(
        X, y_int, paths,
        test_ratio=TEST_RATIO,
        val_ratio_from_remain=VAL_RATIO_FROM_REMAIN,
        seed=SEED
    )

    # Class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weight = dict(zip(np.unique(y_train_int), weights))
    print("Class weights:", class_weight)

    print("\nSplit summary")
    print(f"Train: {len(X_train)}")
    print(f"Val:   {len(X_val)}")
    print(f"Test:  {len(X_test)}")

    # One-hot labels
    y_train = to_categorical(y_train_int, NUM_CLASSES)
    y_val   = to_categorical(y_val_int, NUM_CLASSES)

    train_ds = make_tf_dataset(X_train, y_train, batch_size=BATCH_SIZE, training=True)
    val_ds   = make_tf_dataset(X_val, y_val, batch_size=BATCH_SIZE, training=False)

    # Save split files
    split_dir = os.path.join(OUT_DIR, "splits")
    ensure_dir(split_dir)
    pd.DataFrame({"path": p_train, "label": y_train_int}).to_csv(os.path.join(split_dir, "train.csv"), index=False)
    pd.DataFrame({"path": p_val,   "label": y_val_int}).to_csv(os.path.join(split_dir, "val.csv"), index=False)
    pd.DataFrame({"path": p_test,  "label": y_test_int}).to_csv(os.path.join(split_dir, "test.csv"), index=False)

    all_results = []

    for model_name in ["cnn", "mobilenetv2", "resnet50"]:
        result = train_one_model(
            model_key=model_name,
            train_ds=train_ds,
            val_ds=val_ds,
            X_test=X_test,
            y_test_int=y_test_int,
            save_root=OUT_DIR,
            class_weight=class_weight
        )
        all_results.append(result)

    save_comparison_table(all_results, OUT_DIR)

    print("\nDone.")
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()