#!/usr/bin/env python
"""BCI Competition IV Dataset 2a / 2b 전처리 + label 검증 (MNE ≥ 1.9)"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Sequence, Mapping

import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# ────────── DSP utils ──────────

def butter_bandpass(x: np.ndarray, low: float, high: float, fs: int, order: int = 4):
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def exp_moving_standardize(x, alpha=0.001, eps=1e-4):
    mean = np.zeros(x.shape[:-1]); var = np.ones_like(mean); out = np.empty_like(x)
    for t in range(x.shape[-1]):
        mean = (1 - alpha) * mean + alpha * x[..., t]
        var = (1 - alpha) * var + alpha * (x[..., t] - mean) ** 2
        out[..., t] = (x[..., t] - mean) / np.sqrt(var + eps)
    return out.astype("float32")

# ────────── .mat loader ──────────

def load_mat(mat_path: Path, mapping: Mapping[int,int]|None):
    print(f"[DEBUG] Loading .mat file: {mat_path}")
    if not mat_path.exists():
        print("[DEBUG] .mat file does not exist.")
        return None
    mat = loadmat(mat_path)
    print(f"[DEBUG] Keys in mat file: {mat.keys()}")
    arr = mat["classlabel"].ravel().astype("int64")
    print(f"[DEBUG] Raw labels from .mat: {arr[:10]}")
    if mapping:
        lut = np.arange(max(mapping) + 1, dtype="int64")
        for k, v in mapping.items():
            lut[k] = v
        arr = lut[arr]
    print(f"[DEBUG] Remapped labels: {np.unique(arr)}")
    return arr

# ────────── preprocess single GDF ──────────

def preprocess_gdf(path: Path, tmin, tmax, picks, fs, fl, fh, alpha,
                   label_codes: Sequence[int] | None,
                   true_y: np.ndarray | None = None):
    
    print(f"\n[INFO] Preprocessing {path.name}: picks={picks}, tmin={tmin}, tmax={tmax}, fs={fs}")
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    print(f"[DEBUG] Raw loaded: sfreq={raw.info['sfreq']}, channels={raw.ch_names[:5]}...")

    # 강제 rename
    official_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    if all(ch.startswith("EEG") for ch in raw.ch_names[:22]):
        mapping = {raw.ch_names[i]: official_names[i] for i in range(22)}
        raw.rename_channels(mapping)
        print(f"[DEBUG] Channels renamed to official names: {raw.ch_names[:5]}...")

    if len(set(raw.ch_names)) == 1:
        raw.rename_channels({ch: f"Ch{idx+1:02d}" for idx, ch in enumerate(raw.ch_names)})

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")
        print(f"[DEBUG] Resampled to fs={fs}")

    # 이벤트 매핑 시도
    print(f"[DEBUG] Raw annotations: {np.unique(raw.annotations.description)}")
    try:
        if label_codes:
            event_id = {str(c): c for c in label_codes}
        else:
            # ✅ test 세션인 경우, '783'이 trial start 이벤트임
            event_id = {"783": 783}
        events, _ = mne.events_from_annotations(raw, event_id, verbose=False)
    except Exception as e:
        print(f"[ERROR] Failed to extract events: {e}")
        print(f"[DEBUG] Available annotations: {np.unique(raw.annotations.description)}")
        return np.zeros((0, len(picks), int((tmax - tmin) * fs))), np.zeros((0,), dtype=np.int64)

    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, picks=picks,
                        baseline=None, preload=True, event_repeated="drop", verbose=False)
    data = epochs.get_data()
    print(f"[DEBUG] Epochs → data {data.shape}")
    X = exp_moving_standardize(butter_bandpass(data, fl, fh, fs), alpha)

    if true_y is not None:
        print(f"[DEBUG] true_y shape: {true_y.shape}, values: {np.unique(true_y)}")
        n = min(len(true_y), len(epochs))
        true_y = true_y[:n]
        X = X[:n]
        y = true_y.astype("int64")
    else:
        y = epochs.events[:, -1] if label_codes else np.full(len(epochs), -1, dtype="int64")

    # 라벨을 0부터 시작하는 index로 변환
    remap = {769:0,770:1,771:2,772:3}
    y = np.vectorize(remap.get)(y)
    print(f"[DEBUG] Final shapes X:{X.shape}, y:{y.shape}, unique y:{np.unique(y)}")
    return X, y

# ────────── main ──────────

def run(a):
    root = Path(a.root).expanduser()
    true_dir = root.parent / "true_labels"
    out = Path(a.out).expanduser(); out.mkdir(exist_ok=True)
    fs, fl, fh, alpha = a.fs, a.fl, a.fh, a.alpha

    if a.dataset == "2a":
        subs = a.subjects or [f"A{str(i).zfill(2)}" for i in range(1, 10)]
        train_sfx, test_sfx = ["T"], ["E"]
        eeg_22 = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]
        picks = eeg_22
        tseg = (2, 6)
        codes = [769, 770, 771, 772]
        mapping = {1: 769, 2: 770, 3: 771, 4: 772}
    else:
        subs = a.subjects or [f"B{str(i).zfill(2)}" for i in range(1, 10)]
        train_sfx, test_sfx = ["01T", "02T", "03T"], ["04E", "05E"]
        picks, tseg = "eeg", (3, 7)
        codes = [769, 770]
        mapping = {1: 769, 2: 770}

    for subj in subs:
        print(f"\n[RUN] Subject {subj}")
        sd = out / subj
        sd.mkdir(exist_ok=True)
        Xtr, ytr, Xte, yte = [], [], [], []

        for s in train_sfx:
            print(f"[RUN] Loading train session {subj}{s}")
            mat = load_mat(true_dir / f"{subj}{s}.mat", mapping)
            mat0 = mat if mat is None else np.vectorize({769:0,770:1,771:2,772:3}.get)(mat)
            X, y = preprocess_gdf(root / f"{subj}{s}.gdf", *tseg, picks, fs, fl, fh, alpha, codes, mat)
            print(f"[RUN] Train {subj}{s}: X.shape={X.shape}, y.shape={y.shape}")
            if mat is not None:
                n = min(len(mat), len(y))
                diff = np.where(mat[:n] != y[:n])[0]
                if diff.size:
                    print(f"{subj}{s}: {diff.size} label mismatch (e.g. {diff[:10]})", file=sys.stderr)
            Xtr.append(X); ytr.append(y)

        for s in test_sfx:
            print(f"[RUN] Loading test session {subj}{s}")
            mat = load_mat(true_dir / f"{subj}{s}.mat", mapping)
            mat0 = mat if mat is None else np.vectorize({769:0,770:1,771:2,772:3}.get)(mat)
            X, y = preprocess_gdf(
                root / f"{subj}{s}.gdf",
                *tseg, picks, fs, fl, fh, alpha,
                label_codes=None,  # ✅ 주의! test는 783으로만 자름
                true_y=mat0         # ✅ test는 .mat label을 사용
            )
            Xte.append(X)
            yte.append(y)

        np.save(sd / "train_X.npy", np.concatenate(Xtr))
        np.save(sd / "train_y.npy", np.concatenate(ytr))
        np.save(sd / "test_X.npy",  np.concatenate(Xte))
        np.save(sd / "test_y.npy",  np.concatenate(yte))
    print("\nDone.")

# ────────── CLI ──────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser("BCI IV preprocessing + label check")
    pa.add_argument("--dataset", choices=["2a", "2b"], required=True)
    pa.add_argument("--root", required=True)
    pa.add_argument("--out", default="./preprocessed")
    pa.add_argument("--subjects", nargs="+")
    pa.add_argument("--fs", type=int, default=250)
    pa.add_argument("--fl", type=float, default=8.0)
    pa.add_argument("--fh", type=float, default=32.0)
    pa.add_argument("--alpha", type=float, default=0.001)
    run(pa.parse_args())
