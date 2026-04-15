"""
Project: LiDAR Fingerprint Localization using ConvMLP pose regression
File: preprocess.py

Description:
This script is to clean the collected csv data, apply the augmentaions, normalization and store them so that it can be used in training and inferencing. 

Author: Harsha Keladi Ganapathi
Affiliation: Robotics Lab,
             University of New Haven, CT

License: MIT License
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import re

warnings.filterwarnings("ignore")

RAW_DIR = "dataset/raw/"
OUTPUT_DIR = "dataset/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHYSICAL_MAX_RANGE = 3.5

# ============================================================
# LOAD ALL CSV FILES
# ============================================================
def load_all_csv(raw_folder=RAW_DIR):
    files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(raw_folder, f))
            df["source_file"] = f
            dfs.append(df)
            print(f"Loaded {f} ({len(df)} samples)")
        except Exception as e:
            print(f"Skipping corrupted file: {f} ({e})")

    if not dfs:
        raise RuntimeError("No CSV files found in dataset/raw/")

    return pd.concat(dfs, ignore_index=True)

# ============================================================
# THETA ENCODING
# ============================================================
def encode_theta(df):
    if "theta" in df.columns:
        df["theta_sin"] = np.sin(np.deg2rad(df["theta"]))
        df["theta_cos"] = np.cos(np.deg2rad(df["theta"]))
        df = df.drop(columns=["theta"])
    return df

# ============================================================
# CLEAN DATASET
# ============================================================
def clean_dataset(df):

    # Replace empty strings with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # ========================================================
    # >>> NEW: Remove unwanted orientation columns
    # ========================================================
    drop_cols = ["yaw", "sin_yaw", "cos_yaw"]
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"Dropped unwanted columns: {existing}")

    # Identify LiDAR columns EXACTLY r<digits>
    scan_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"r\d+", c)],
        key=lambda s: int(s[1:])
    )

    # Drop rows missing x or y ONLY
    df = df.dropna(subset=["x", "y"], how="any")

    
    if scan_cols:
        df[scan_cols] = df[scan_cols].replace(100, PHYSICAL_MAX_RANGE)

    if "max_dist" in df.columns:
        df["max_dist"] = df["max_dist"].replace(100, PHYSICAL_MAX_RANGE)

    # Fill NaN in lidar
    if scan_cols:
        df[scan_cols] = df[scan_cols].fillna(0.0)

    # HARD CLAMP REAL HARDWARE LIDAR VALUES
    if scan_cols:
        df[scan_cols] = df[scan_cols].clip(0.0, PHYSICAL_MAX_RANGE)

    # Clamp max_dist too
    if "max_dist" in df.columns:
        df["max_dist"] = df["max_dist"].clip(0.0, PHYSICAL_MAX_RANGE)

    # Fill NaN in other features
    non_scan = [c for c in df.columns if c not in scan_cols + ["source_file"]]
    df[non_scan] = df[non_scan].fillna(0.0)

    df = df.drop_duplicates()

    print(f"Detected {len(scan_cols)} LiDAR rays in cleaning stage.")
    return df

# ============================================================
# AUGMENTATION
# ============================================================
def augment_lidar(scan):
    scan = scan.copy()
    noise = np.random.normal(0, 0.02 * scan.mean(), size=scan.shape)
    scan = scan + noise

    drop_count = int(0.02 * len(scan))
    if drop_count > 0:
        idx = np.random.choice(len(scan), drop_count, replace=False)
        scan[idx] = 0.0

    return scan

def jitter_xy(x, y):
    return (
        x + np.random.uniform(-0.02, 0.02),
        y + np.random.uniform(-0.02, 0.02)
    )

# ============================================================
# MAIN PREPROCESSING
# ============================================================
def preprocess():

    print("\n=== LOADING CSV FILES ===")
    df = load_all_csv(RAW_DIR)

    print("\n=== CLEANING DATASET ===")
    df = clean_dataset(df)
    print("Samples after cleaning:", len(df))

    print("\n=== SAVING CLEANED CSV ===")
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_dataset.csv"), index=False)

    #print("\n=== ENCODING THETA ===")
    #df = encode_theta(df)

    target_cols = ["x", "y"]

    scan_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"r\d+", c)],
        key=lambda s: int(s[1:])
    )

    feature_cols = [
        c for c in df.columns
        if c not in target_cols + ["source_file", "id"] + scan_cols
    ]

    full_feature_cols = feature_cols + scan_cols

    print(f"Num LiDAR rays     = {len(scan_cols)}")
    print(f"Non-scan features  = {len(feature_cols)}")
    print(f"Total feature dims = {len(full_feature_cols)}")

    X_list, y_list = [], []

    print("\n=== AUGMENTING DATA ===")

    for idx, row in df.iterrows():

        extra = row[feature_cols].values.astype(np.float32)
        scan = row[scan_cols].values.astype(np.float32)

        x, y = float(row["x"]), float(row["y"])

        X_list.append(np.concatenate([extra, scan]))
        y_list.append([x, y])

        for _ in range(3):
            scan_aug = augment_lidar(scan)
            x_j, y_j = jitter_xy(x, y)

            X_list.append(np.concatenate([extra, scan_aug]))
            y_list.append([x_j, y_j])

        if idx % 3000 == 0:
            print(f"Processed {idx}/{len(df)}...")

    X = np.array(X_list)
    Y = np.array(y_list, dtype=np.float32)

    print(f"\nOriginal samples: {len(df)}")
    print(f"After augmentation: {len(X)}")

    print("\n=== NORMALIZING FEATURES ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_scaled)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), Y)
    pickle.dump(scaler, open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb"))

    meta = {
        "num_original": len(df),
        "num_final": len(X),
        "feature_dims": X.shape[1],
        "feature_cols": feature_cols,
        "scan_cols": scan_cols,
        "augmentations_per_sample": 3,
        "noise": "Gaussian + dropout + coord jitter",
    }

    with open(os.path.join(OUTPUT_DIR, "preprocessor_log.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("\n=== PREPROCESSING COMPLETE ===")
    print("Saved inside:", OUTPUT_DIR)

if __name__ == "__main__":
    preprocess()

