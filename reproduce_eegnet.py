"""EEGNet A01 reproduction on BCI Competition IV-2a.

The EEGNet architecture is imported directly from the original ARL
EEGModels repository:
https://github.com/vlawhern/arl-eegmodels

This script reproduces the experiment documented in the repository's
README/METHODS files. It is designed for Google Colab CPU.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.callbacks import ModelCheckpoint

ROOT = Path("/content/eegnet_reproduction")
DATA_DIR = ROOT / "BCICIV_2a"
LABEL_DIR = ROOT / "true_labels"
AUTHOR_REPO = ROOT / "arl-eegmodels"
RESULT_DIR = ROOT / "results"
for d in (DATA_DIR, LABEL_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def sh(cmd: str):
    subprocess.run(cmd, shell=True, check=True)


def ensure_author_code():
    if not AUTHOR_REPO.exists():
        sh(f"git clone -q https://github.com/vlawhern/arl-eegmodels.git {AUTHOR_REPO}")
    sys.path.insert(0, str(AUTHOR_REPO))


def exponential_moving_standardize(data, factor_new=1e-3, init_block_size=1000, eps=1e-4):
    x = pd.DataFrame(data.T)
    meaned = x.ewm(alpha=factor_new).mean()
    demeaned = x - meaned
    square_ewmed = (demeaned * demeaned).ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.asarray(square_ewmed)))
    standardized = np.asarray(standardized)
    if init_block_size is not None:
        x_np = np.asarray(x)
        init_mean = np.mean(x_np[:init_block_size], axis=0, keepdims=True)
        init_std = np.std(x_np[:init_block_size], axis=0, keepdims=True)
        init_standardized = (x_np[:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[:init_block_size] = init_standardized
    return standardized.T


def preprocess(raw):
    # BCI IV-2a: first 22 channels are EEG; last 3 are EOG.
    raw.pick(raw.ch_names[:22])
    raw.resample(128, npad="auto", verbose=False)
    raw.filter(l_freq=4.0, h_freq=40.0, fir_design="firwin", verbose=False)
    raw._data = exponential_moving_standardize(
        raw.get_data() * 1e6,
        factor_new=1e-3,
        init_block_size=1000,
    )
    return raw


def load_train_session(path):
    raw = preprocess(mne.io.read_raw_gdf(path, preload=True, verbose=False))
    events, _ = mne.events_from_annotations(raw, verbose=False)
    mi_event_id = {"left_hand": 7, "right_hand": 8, "feet": 9, "tongue": 10}
    epochs = mne.Epochs(
        raw, events, event_id=mi_event_id,
        tmin=0.5, tmax=2.5 - 1/128,
        baseline=None, preload=True, reject_by_annotation=True, verbose=False,
    )
    X = epochs.get_data().astype(np.float32)[..., np.newaxis]
    label_map = {7: 0, 8: 1, 9: 2, 10: 3}
    y = np.asarray([label_map[v] for v in epochs.events[:, 2]], dtype=np.int64)
    return X, y


def load_eval_session(gdf_path, mat_path):
    raw = preprocess(mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False))
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    trial_starts = events[events[:, 2] == event_id["768"]]
    epochs = mne.Epochs(
        raw, trial_starts, event_id={"trial_start": event_id["768"]},
        tmin=2.5, tmax=4.5 - 1/128,
        baseline=None, preload=True, reject_by_annotation=True, verbose=False,
    )
    X = epochs.get_data().astype(np.float32)[..., np.newaxis]
    y = loadmat(mat_path)["classlabel"].reshape(-1).astype(np.int64) - 1
    return X, y


def build_eegnet():
    ensure_author_code()
    from EEGModels import EEGNet
    model = EEGNet(
        nb_classes=4, Chans=22, Samples=256,
        dropoutRate=0.5, kernLength=32,
        F1=8, D=2, F2=16, norm_rate=0.25,
        dropoutType="Dropout",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ensure_author_code()

    # Download official dataset if needed.
    if not (Path("/content/BCICIV_2a_gdf.zip")).exists():
        sh("wget -q -O /content/BCICIV_2a_gdf.zip https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip")
    if not (DATA_DIR / "A01T.gdf").exists():
        sh("unzip -q -o /content/BCICIV_2a_gdf.zip -d /content/BCICIV_2a")

    # Download official evaluation labels if needed.
    if not (Path("/content/true_labels.zip")).exists():
        sh("wget -q -O /content/true_labels.zip https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip")
    if not (LABEL_DIR / "A01E.mat").exists():
        sh("unzip -q -o /content/true_labels.zip -d /content/true_labels")

    print("Loading A01T...")
    X_T, y_T = load_train_session(DATA_DIR / "A01T.gdf")
    print("Loading A01E...")
    X_E, y_E = load_eval_session(DATA_DIR / "A01E.gdf", LABEL_DIR / "A01E.mat")
    print("Training data:", X_T.shape)
    print("Evaluation data:", X_E.shape)

    folds = [
        (np.concatenate([np.arange(96,192), np.arange(192,288)]), np.arange(0,96)),
        (np.concatenate([np.arange(192,288), np.arange(0,96)]), np.arange(96,192)),
        (np.concatenate([np.arange(0,96), np.arange(96,192)]), np.arange(192,288)),
    ]

    out = []
    for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n========== FOLD {fold_no}/3 ==========")
        tf.keras.backend.clear_session()
        gc.collect()
        model = build_eegnet()
        print("Parameters:", model.count_params())
        ckpt = RESULT_DIR / f"eegnet_A01_fold{fold_no}.weights.h5"
        callback = ModelCheckpoint(str(ckpt), monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True)
        start = time.time()
        hist = model.fit(
            X_T[train_idx], y_T[train_idx],
            batch_size=64, epochs=500,
            validation_data=(X_T[val_idx], y_T[val_idx]),
            callbacks=[callback], verbose=2,
        )
        minutes = (time.time() - start) / 60.0
        model.load_weights(str(ckpt))
        _, test_acc = model.evaluate(X_E, y_E, verbose=0)
        fold = {
            "fold": fold_no,
            "parameters": int(model.count_params()),
            "best_validation_accuracy_percent": float(max(hist.history["val_accuracy"]) * 100),
            "evaluation_accuracy_percent": float(test_acc * 100),
            "training_time_minutes": float(minutes),
        }
        print(fold)
        out.append(fold)
        del model
        gc.collect()

    mean_acc = float(np.mean([r["evaluation_accuracy_percent"] for r in out]))
    std_acc = float(np.std([r["evaluation_accuracy_percent"] for r in out]))
    total_time = float(sum(r["training_time_minutes"] for r in out))
    summary = {
        "subject": "A01",
        "folds": out,
        "mean_evaluation_accuracy_percent": mean_acc,
        "std_evaluation_accuracy_percent": std_acc,
        "total_training_time_minutes": total_time,
    }
    with open(RESULT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n===== FINAL RESULT =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
