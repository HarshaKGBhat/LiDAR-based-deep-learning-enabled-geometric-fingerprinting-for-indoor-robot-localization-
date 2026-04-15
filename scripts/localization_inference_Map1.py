"""
Project: LiDAR Fingerprint Localization using ConvMLP pose regression
File: localization_inference.py

Description:
Map1 ROS2 bag replay of end-to-end inference pipeline for LiDAR-based ConvMLP pose regression including feature extraction, scaling, ConvMLP prediction,and recursive EKF-based smoothing. 

Author: Harsha Keladi Ganapathi
Affiliation: Robotics Lab,
             University of New Haven, CT

License: MIT License
"""




#!/usr/bin/env python3

import os
import math
import time
import json
import csv
from collections import deque

import numpy as np
from numpy.fft import rfft
from scipy.stats import skew, kurtosis, entropy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from tf2_ros import Buffer, TransformListener

import psutil  # for memory usage

# Optional PyTorch for deep models + MC dropout
USE_TORCH = True
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    USE_TORCH = False

# ---------------- CONFIG ----------------
MODEL_TYPE = os.environ.get("MODEL_TYPE", "pytorch")   # "sklearn" or "pytorch"
SKLEARN_MODEL_PATH = os.environ.get("SKLEARN_MODEL_PATH", "models/SklearnMLP.pkl")
PYTORCH_MODEL_PATH = os.environ.get("PYTORCH_MODEL_PATH", "models/ConvMLP_Best_1.pt")

SCALER_PATH = "dataset/processed/scaler.pkl"
PREPROC_LOG = "dataset/processed/preprocessor_log.json"

# training / preprocess used 360 bins (r0..r359)
N_BINS_FULL = 360
PHYSICAL_MAX_RANGE = 3.5  # real hardware lidar max 

# MC dropout params (only used for PyTorch)
ENABLE_MC = bool(int(os.environ.get("ENABLE_MC", "0")))  # 0/1
MC_T = int(os.environ.get("MC_T", "20"))

# Logging / output dirs
OUT_DIR = "inference_log"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Error / Kidnap experiment config ---
SUCCESS_THRESH_M = 0.30        # success if error < 0.30 m
KIDNAP_ERROR_THRESH_M = 1.0    # consider "kidnapped" if ML error > 1.0 m
KIDNAP_STABLE_FRAMES = 10      # frames below SUCCESS_THRESH_M to count as recovered

def interpolate_to_360(scan):
    scan = np.asarray(scan, dtype=float)
    n = len(scan)
    if n == N_BINS_FULL:
        return scan
    old_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    new_angles = np.linspace(0.0, 2.0 * np.pi, N_BINS_FULL, endpoint=False)
    return np.interp(new_angles, old_angles, scan)


# ---------------------------
# Feature extractor (same as training / collector)
# ---------------------------
def extract_optimized_features(scan, max_range):
    d = np.asarray(scan, dtype=np.float64)
    if len(d) != N_BINS_FULL:
        d = interpolate_to_360(d)
    d = np.clip(d, 0.0, PHYSICAL_MAX_RANGE)
    max_range = PHYSICAL_MAX_RANGE

    std_val = float(np.std(d))
    min_val = float(np.min(d))
    max_val = float(np.max(d))
    skew_val = float(skew(d))
    kurt_val = float(kurtosis(d))

    diffs = np.abs(np.diff(d))
    roughness = float(np.mean(diffs))
    jump_rate = float(np.sum(diffs > 0.1) / max(1, len(diffs)))

    hist_counts, _ = np.histogram(d, bins=20, range=(0, max_range))
    ent = float(entropy(hist_counts + 1e-6))

    fft_vals = np.abs(rfft(d))
    try:
        fft_std = float(np.std(np.sort(fft_vals)[-10:]))
    except Exception:
        fft_std = float(np.std(fft_vals))

    angles = np.linspace(0, 2 * np.pi, N_BINS_FULL, endpoint=False)
    X = d * np.cos(angles)
    Y = d * np.sin(angles)
    x_span = float(np.ptp(X))
    y_span = float(np.ptp(Y))
    aspect_ratio = float(max(x_span, y_span) / (min(x_span, y_span) + 1e-6))

    diffs_circ = np.abs(np.diff(d, prepend=d[-1]))
    num_clusters = int(np.sum(diffs_circ > 1.0))

    return {
        "std_dist": std_val,
        "min_dist": min_val,
        "max_dist": max_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "roughness": roughness,
        "jump_rate": jump_rate,
        "entropy_dist": ent,
        "fft_std": fft_std,
        "aspect_ratio": aspect_ratio,
        "num_clusters": num_clusters,
        "clearance_proxy": min_val
    }


# ---------------------------
# Quaternion to yaw
# ---------------------------
def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


# ---------------------------
# If using PyTorch: defining same architecture as training
# ---------------------------
if USE_TORCH:
    class ConvMLPNet(nn.Module):
        def __init__(self, input_dim, dropout_p=0.3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_p),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_p),
            )
            # After two MaxPool1d(2), length is input_dim / 4
            reduced = input_dim // 4
            self.mlp = nn.Sequential(
                nn.Linear(64 * reduced, 512),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

        def forward(self, x):
            # x: (B, L)
            x = x.unsqueeze(1)  # (B, 1, L)
            x = self.conv(x)
            x = x.flatten(1)
            return self.mlp(x)
#----------------------------
# Recursive EKF
# ---------------------------
class RecursiveEKF:
    def __init__(self):

        # noise 
        self.Q = np.diag([0.02, 0.02, 0.01])
        self.R = np.diag([
            0.2, 0.2,
            0.3, 0.3,
            0.2, 0.2,
            0.05, 0.05,
            0.02
        ])

        self.P = np.eye(3)
        self.x = None

        self.hist_x = deque(maxlen=10)
        self.hist_y = deque(maxlen=10)

    def update(self, dt, odomx, odomy, mlx, mly, yaw, v, omega):

        if self.x is None:
            self.x = np.array([odomx, odomy, yaw])
            return self.x

        # recursive shaping
        avgx = 0.5 * (odomx + self.x[0])
        avgy = 0.5 * (odomy + self.x[1])

        self.hist_x.append(avgx)
        self.hist_y.append(avgy)

        smoothx = np.mean(self.hist_x)
        smoothy = np.mean(self.hist_y)

        z = np.array([
            odomx, odomy,
            mlx, mly,
            avgx, avgy,
            smoothx, smoothy,
            yaw
        ])

        F = np.array([
            [1, 0, -dt*v*np.sin(self.x[2])],
            [0, 1,  dt*v*np.cos(self.x[2])],
            [0, 0, 1]
        ])

        H = np.array([
            [1,0,0],[0,1,0],
            [1,0,0],[0,1,0],
            [1,0,0],[0,1,0],
            [1,0,0],[0,1,0],
            [0,0,1]
        ])

        # predict
        x_pred = np.array([
            self.x[0] + dt*v*np.cos(self.x[2]),
            self.x[1] + dt*v*np.sin(self.x[2]),
            self.x[2] + dt*omega
        ])

        P_pred = F @ self.P @ F.T + self.Q

        # update
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ (z - H @ x_pred)

        self.P = np.linalg.inv(
            np.linalg.inv(P_pred) + H.T @ np.linalg.inv(self.R) @ H
        )

        return self.x

# ---------------------------
# Node
# ---------------------------
class LocalizationInferenceLive(Node):
    def __init__(self):
        super().__init__("localization_inference_live")

        # TF buffer/listener for SLAM GT (map -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # /scan with BEST_EFFORT QoS
        scan_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(LaserScan, "/scan", self.scan_cb, scan_qos)

        # /odom for odom trajectory
        self.create_subscription(Odometry, "/odom", self.odom_cb, 50)

        # Publisher (GT vs ML for external logging/EKF)
        self.eval_pub = self.create_publisher(Float32MultiArray, "/localization_eval", 10)

        # State
        self.last_scan = None
        self.last_scan_max = PHYSICAL_MAX_RANGE

        self.last_pose_gt = None      # (x, y, yaw) from map->base_link
        self.last_pose_odom = None    # (x, y, yaw) RAW ODOM (odom frame)
        self.last_odom_in_map = None  # (x, y, yaw_map) odom expressed in map
        self.last_scan_time = None    # ROS timestamp (float seconds) from /scan
	

        # calling Recursive EKF
        self.ekf = RecursiveEKF()
        self.ekf_history = []
        self.prev_time = None
        self.prev_pose = None

        # Metrics storage (timing / memory)
        self.feature_times = []
        self.scale_times = []
        self.infer_times = []
        self.total_times = []
        self.mem_mb_list = []

        # Error metrics storage
        self.error_threshold = SUCCESS_THRESH_M
        self.ml_errors = []     # Euclidean error ML vs SLAM
        self.odom_errors = []   # Euclidean error Odom vs SLAM
        self.ml_success_count = 0
        self.odom_success_count = 0

        # Kidnap / recovery tracking (based on ML error)
        self.kidnapped = False
        self.kidnap_start_time = None
        self.kidnap_events = 0
        self.kidnap_recovery_times = []    # list of recovery durations (s)
        self.recent_ml_errors = deque(maxlen=KIDNAP_STABLE_FRAMES)

        # For memory usage
        self.proc = psutil.Process(os.getpid())

        # Timestamps and paths
        self.ts = time.strftime("%Y%m%d_%H%M%S")
        self.scan_csv_path = os.path.join(OUT_DIR, f"scans_{self.ts}.csv")
        self.scan_csv_fh = open(self.scan_csv_path, "w", newline="")
        self.scan_writer = None  # will initialize header later

        self.infer_log_path = os.path.join(OUT_DIR, f"inference_{self.ts}.csv")
        self.metrics_path = os.path.join(OUT_DIR, f"inference_metrics_{self.ts}.json")

        # Inference CSV header (metrics + predictions)
        with open(self.infer_log_path, "w") as fh:
            fh.write(
                "time,x_true,y_true,x_pred,y_pred,"
                "x_std,y_std,"
                "feat_time_ms,scale_time_ms,infer_time_ms,total_time_ms,"
                "memory_mb,fps\n"
            )
        """
        # === Unified EKF CSV ===
        self.ekf_csv_path = os.path.join(OUT_DIR, f"ekf_data_{self.ts}.csv")
        self.ekf_fh = open(self.ekf_csv_path, "w", newline="")
        self.ekf_writer = csv.writer(self.ekf_fh)
        self.ekf_writer.writerow([
            "time",
            "slam_x", "slam_y", "slam_yaw",
            "odom_x", "odom_y", "odom_yaw",
            "ml_x", "ml_y"
        ])
        """
        """
        # === Errors CSV ===
        self.errors_csv_path = os.path.join(OUT_DIR, f"errors_{self.ts}.csv")
        self.errors_fh = open(self.errors_csv_path, "w", newline="")
        self.errors_writer = csv.writer(self.errors_fh)
        self.errors_writer.writerow([
            "time",
            "slam_x", "slam_y", "slam_yaw",
            "ml_x", "ml_y",
            "odom_x", "odom_y", "odom_yaw",
            "ml_error", "odom_error",
            "ml_error_x", "ml_error_y",
            "odom_error_x", "odom_error_y",
            "error_ratio",
            "success_flag_ml",
            "success_flag_odom"
        ])
        """
        # Load scaler
        self.scaler = None
        if os.path.exists(SCALER_PATH):
            try:
                import pickle
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.get_logger().info(f"Loaded scaler: {SCALER_PATH}")
            except Exception as e:
                self.get_logger().warning(f"Failed loading scaler: {e}")
        else:
            self.get_logger().warning("Scaler not found; continuing without scaler (not recommended).")

        # Load feature column order (feature_cols + scan_cols from preproc log)
        self.feature_columns = None
        if os.path.exists(PREPROC_LOG):
            try:
                with open(PREPROC_LOG, "r") as f:
                    log = json.load(f)

                if "feature_columns" in log:
                    self.feature_columns = log["feature_columns"]
                    self.get_logger().info(
                        f"Loaded feature_columns from log ({len(self.feature_columns)} cols)."
                    )
                else:
                    base = log.get("feature_cols", [])
                    scans = log.get("scan_cols", [])
                    self.feature_columns = base + scans
                    self.get_logger().warn(
                        "feature_columns missing in preprocessor log — reconstructed as "
                        "'feature_cols + scan_cols'. Using reconstructed feature order."
                    )
            except Exception as e:
                self.get_logger().warning(f"Failed to load preproc log: {e}")
        else:
            self.get_logger().warning("preprocessor_log.json not found - feature ordering unknown.")

        # ------------------------------
        # Load model depending on type
        # ------------------------------
        self.model = None
        self.pytorch_model = None

        if MODEL_TYPE == "sklearn":
            try:
                import pickle
                if not os.path.exists(SKLEARN_MODEL_PATH):
                    raise FileNotFoundError(SKLEARN_MODEL_PATH)
                with open(SKLEARN_MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                self.get_logger().info(f"Loaded sklearn model: {SKLEARN_MODEL_PATH}")
            except Exception as e:
                self.get_logger().error(f"Failed to load sklearn model: {e}")
                raise
        elif MODEL_TYPE == "pytorch":
            if not USE_TORCH or torch is None:
                self.get_logger().error("PyTorch not available in environment.")
                raise RuntimeError("PyTorch missing")
            try:
                if not os.path.exists(PYTORCH_MODEL_PATH):
                    raise FileNotFoundError(PYTORCH_MODEL_PATH)

                # Determine input dim from feature_columns or scaler
                if self.feature_columns is not None:
                    input_dim = len(self.feature_columns)
                else:
                    if self.scaler is None or not hasattr(self.scaler, "mean_"):
                        raise RuntimeError(
                            "Cannot determine input dim: feature_columns missing and scaler has no mean_."
                        )
                    input_dim = int(self.scaler.mean_.shape[0])
                    self.get_logger().warn(
                        f"feature_columns missing → using scaler dimension as fallback (input_dim={input_dim})"
                    )

                self.pytorch_model = ConvMLPNet(input_dim=input_dim, dropout_p=0.3)
                state = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")

                if isinstance(state, dict) and "state_dict" in state:
                    sd = state["state_dict"]
                else:
                    sd = state

                try:
                    self.pytorch_model.load_state_dict(sd)
                except Exception:
                    self.pytorch_model.load_state_dict(sd, strict=False)

                self.pytorch_model.eval()
                self.get_logger().info(f"Loaded PyTorch model: {PYTORCH_MODEL_PATH}")
            except Exception as e:
                self.get_logger().error(f"Failed to load PyTorch model: {e}")
                raise
        else:
            self.get_logger().error(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
            raise RuntimeError("Unknown MODEL_TYPE")

        # Live plot state
        self.odom_history = deque(maxlen=2000)  # odom trail (in map frame for viz)
        self.gt_history = []                    # SLAM GT (map)
        self.pred_history = []                  # ML predictions (map)

        # matplotlib interactive
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(figsize=(7, 6))
            self.last_plot_time = time.time()
        except Exception as e:
            self.plt = None
            self.get_logger().warning(f"Matplotlib not available: {e}")

        self.get_logger().info("Localization inference LIVE node initialized.")

    # --------------------------
    # ODOM -> used for odom trajectory + EKF raw odom
    # --------------------------
    def odom_cb(self, msg: Odometry):

        # Wait until Cartographer publishes map->base_link
        if not self.tf_buffer.can_transform("map", "base_link", Time()):
            return

        # Raw odom pose (odom frame)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw_odom = quat_to_yaw(q)
        ox, oy = float(p.x), float(p.y)

        # Save RAW odom for EKF CSV (Option A)
        self.last_pose_odom = (ox, oy, yaw_odom)

        try:
            # SLAM pose: map -> base_link
            tf_map_base = self.tf_buffer.lookup_transform(
                "map", "base_link",
                Time(),
                timeout=Duration(seconds=0.05)
            )

            bx = tf_map_base.transform.translation.x
            by = tf_map_base.transform.translation.y
            qmb = tf_map_base.transform.rotation
            yaw_mb = quat_to_yaw(qmb)

            # Rotation matrices
            c_odom = math.cos(yaw_odom)
            s_odom = math.sin(yaw_odom)
            R_odom = np.array([[c_odom, -s_odom],
                               [s_odom,  c_odom]])

            c_map = math.cos(yaw_mb)
            s_map = math.sin(yaw_mb)
            R_map = np.array([[c_map, -s_map],
                              [s_map,  c_map]])

            # Inverse(odom->base_link)
            R_odom_inv = R_odom.T
            t_odom_inv = -R_odom_inv @ np.array([ox, oy])

            # map->odom
            R_map_odom = R_map @ R_odom_inv
            t_map_odom = np.array([bx, by]) + R_map @ t_odom_inv

            # Transform odom pose into map frame (for plotting only)
            odom_in_map = R_map_odom @ np.array([ox, oy]) + t_map_odom
            xm, ym = odom_in_map[0], odom_in_map[1]
            oyaw_map = yaw_odom + yaw_mb
            self.last_odom_in_map = (xm, ym, oyaw_map)

            # Save for plotting
            self.odom_history.append((xm, ym))

        except Exception as e:
            self.get_logger().warn(f"TF failed, using raw odom for viz: {e}")
            self.odom_history.append((ox, oy))

        # Plotting (throttled by wall-clock is fine)
        if self.plt is not None and time.time() - self.last_plot_time > 0.6:
            try:
                self._update_plot()
                self.last_plot_time = time.time()
            except Exception as e:
                self.get_logger().warning(f"Plot update error: {e}")

    # --------------------------
    # SCAN + TF → get SLAM GT & run inference Note: we are using the SLAM GT just to compare the ConvMLP result. The developed method does not have any connection with SLAM GT to predict the localization.
    # --------------------------
    def scan_cb(self, msg: LaserScan):
        # ROS timestamp from scan (master time for all logging)
        self.last_scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Convert raw ranges 
        raw = np.asarray(msg.ranges, dtype=float)
        raw = np.nan_to_num(
            raw,
            nan=PHYSICAL_MAX_RANGE,
            posinf=PHYSICAL_MAX_RANGE,
            neginf=0.0
        )
        raw = np.clip(raw, 0.0, PHYSICAL_MAX_RANGE)
        scan_360 = interpolate_to_360(raw)

        self.last_scan_max = PHYSICAL_MAX_RANGE
        self.last_scan = scan_360

        # Get SLAM GT: map -> base_link
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "base_link", Time()
            )
            tx = tf.transform.translation
            q = tf.transform.rotation
            x_gt = float(tx.x)
            y_gt = float(tx.y)
            yaw_gt = quat_to_yaw(q)
            self.last_pose_gt = (x_gt, y_gt, yaw_gt)
            self.gt_history.append((x_gt, y_gt))
        except Exception as e:
            self.get_logger().warn(f"TF map->base_link not ready: {e}")
            return

        if self.last_pose_gt is not None:
            self.run_inference()

    # --------------------------
    
           
    def build_feature_vector(self, ranges, yaw_gt):
    	feats = extract_optimized_features(ranges, self.last_scan_max)
    	vals = {}
    	# ONLY geometric + lidar features
    	for k, v in feats.items():
        	vals[k] = float(v)

    	for i in range(N_BINS_FULL):
        	vals[f"r{i}"] = float(ranges[i])

    	if self.feature_columns:
        	fv = [vals.get(col, 0.0) for col in self.feature_columns]
        	fv = np.array(fv, dtype=np.float64).reshape(1, -1)
        	return fv, vals
    	else:
        # fallback (yaw removed)
        	fv = np.concatenate([
            	np.array(list(feats.values())),
            	ranges
        	])
        	return fv.reshape(1, -1), vals

    
    # --------------------------
    def mc_predict_pytorch(self, fv_scaled_np, T=MC_T):
        if self.pytorch_model is None:
            return None, None
        self.pytorch_model.train()   # enable dropout
        preds = []
        with torch.no_grad():
            Xb = torch.tensor(fv_scaled_np, dtype=torch.float32)
            device = next(self.pytorch_model.parameters()).device
            Xb = Xb.to(device)
            for _ in range(T):
                out = self.pytorch_model(Xb).cpu().numpy()
                preds.append(out)
        self.pytorch_model.eval()
        preds = np.stack(preds, axis=0)   # (T, N, 2)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    # --------------------------
    def run_inference(self):
        if self.last_scan is None or self.last_pose_gt is None:
            return

        # Use ROS scan timestamp as master time
        if self.last_scan_time is not None:
            t_stamp = self.last_scan_time
        else:
            # fallback (should be rare)
            t_stamp = time.time()

        t_total_start = time.perf_counter()

        x_true, y_true, yaw_gt = self.last_pose_gt

        # 1) Feature construction
        t_feat_start = time.perf_counter()
        fv, vals = self.build_feature_vector(self.last_scan, yaw_gt)
        t_feat_end = time.perf_counter()
        feat_time_ms = (t_feat_end - t_feat_start) * 1000.0

        # 2) Scaling
        t_scale_start = time.perf_counter()
        if self.scaler is not None and hasattr(self.scaler, "transform"):
            try:
                fv_scaled = self.scaler.transform(fv)
            except Exception as e:
                self.get_logger().error(f"Scaler transform failed: {e}")
                fv_scaled = fv
        else:
            fv_scaled = fv
        t_scale_end = time.perf_counter()
        scale_time_ms = (t_scale_end - t_scale_start) * 1000.0

        # 3) Model inference
        t_infer_start = time.perf_counter()
        x_pred, y_pred = None, None
        pred_std = None

        try:
            if MODEL_TYPE == "sklearn":
                pred = self.model.predict(fv_scaled)[0]
                x_pred, y_pred = float(pred[0]), float(pred[1])
            else:
                if ENABLE_MC:
                    mean, std = self.mc_predict_pytorch(fv_scaled, T=MC_T)
                    x_pred, y_pred = float(mean[0, 0]), float(mean[0, 1])
                    pred_std = (float(std[0, 0]), float(std[0, 1]))
                else:
                    Xb = torch.tensor(fv_scaled, dtype=torch.float32)
                    device = next(self.pytorch_model.parameters()).device
                    Xb = Xb.to(device)
                    with torch.no_grad():
                        out = self.pytorch_model(Xb).cpu().numpy()[0]
                    x_pred, y_pred = float(out[0]), float(out[1])
        except Exception as e:
            self.get_logger().error(f"Prediction error: {e}")
            return

        t_infer_end = time.perf_counter()
        infer_time_ms = (t_infer_end - t_infer_start) * 1000.0

        # 4) Total time + memory + FPS
        t_total_end = time.perf_counter()
        total_time_ms = (t_total_end - t_total_start) * 1000.0
        mem_mb = self.proc.memory_info().rss / (1024.0 * 1024.0)
        fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

        # store metrics for summary
        self.feature_times.append(feat_time_ms)
        self.scale_times.append(scale_time_ms)
        self.infer_times.append(infer_time_ms)
        self.total_times.append(total_time_ms)
        self.mem_mb_list.append(mem_mb)

        # --------------------------
        # Error metrics: ML vs SLAM, Odom vs SLAM (in map frame)
        # --------------------------
        ml_err_x = x_pred - x_true
        ml_err_y = y_pred - y_true
        e_ml = math.sqrt(ml_err_x ** 2 + ml_err_y ** 2)

        if self.last_odom_in_map is not None:
            oxm, oym, oyaw_map = self.last_odom_in_map
            odom_err_x = oxm - x_true
            odom_err_y = oym - y_true
            e_odom = math.sqrt(odom_err_x ** 2 + odom_err_y ** 2)
        else:
            oxm = oym = oyaw_map = float("nan")
            odom_err_x = odom_err_y = float("nan")
            e_odom = float("nan")

        # record arrays for summary
        self.ml_errors.append(e_ml)
        self.odom_errors.append(e_odom)

        # success flags (Option 3)
        success_flag_ml = int(e_ml < self.error_threshold)
        self.ml_success_count += success_flag_ml

        if math.isfinite(e_odom):
            success_flag_odom = int(e_odom < self.error_threshold)
        else:
            success_flag_odom = 0
        self.odom_success_count += success_flag_odom

        # error ratio: odom / ml (avoid division by zero)
        if e_ml > 1e-6 and math.isfinite(e_odom):
            error_ratio = e_odom / e_ml
        else:
            error_ratio = float("nan")

        # Kidnap detection & recovery tracking (based on ML error)
        self.recent_ml_errors.append(e_ml)
        if (not self.kidnapped) and (e_ml > KIDNAP_ERROR_THRESH_M):
            # enter kidnapped state
            self.kidnapped = True
            self.kidnap_start_time = t_stamp
            self.kidnap_events += 1

        if self.kidnapped:
            if len(self.recent_ml_errors) == KIDNAP_STABLE_FRAMES and all(
                err < self.error_threshold for err in self.recent_ml_errors
            ):
                # recovered
                if self.kidnap_start_time is not None:
                    recovery_time = t_stamp - self.kidnap_start_time
                    self.kidnap_recovery_times.append(recovery_time)
                self.kidnapped = False
                self.kidnap_start_time = None

        # Publish eval (GT vs ML)
        msg = Float32MultiArray()
        msg.data = [x_true, y_true, x_pred, y_pred]
        self.eval_pub.publish(msg)

        # 5) Log metrics + prediction to inference CSV
        x_std_str = ""
        y_std_str = ""
        if pred_std is not None:
            x_std_str = f"{pred_std[0]:.6f}"
            y_std_str = f"{pred_std[1]:.6f}"

        try:
            with open(self.infer_log_path, "a") as fh:
                fh.write(
                    f"{t_stamp:.6f},"
                    f"{x_true:.6f},{y_true:.6f},"
                    f"{x_pred:.6f},{y_pred:.6f},"
                    f"{x_std_str},{y_std_str},"
                    f"{feat_time_ms:.4f},{scale_time_ms:.4f},"
                    f"{infer_time_ms:.4f},{total_time_ms:.4f},"
                    f"{mem_mb:.2f},{fps:.2f}\n"
                )
        except Exception as e:
            self.get_logger().warning(f"Failed to write inference CSV row: {e}")

        # 6) Save raw scan CSV (separate file)
        try:
            if self.scan_writer is None:
                header = ["time", "x_true", "y_true", "yaw_gt", "x_pred", "y_pred"]
                if pred_std is not None:
                    header += ["x_std", "y_std"]
                header += [f"r{i}" for i in range(N_BINS_FULL)]
                self.scan_writer = csv.DictWriter(self.scan_csv_fh, fieldnames=header)
                self.scan_writer.writeheader()

            row = {
                "time": t_stamp,
                "x_true": x_true,
                "y_true": y_true,
                "yaw_gt": yaw_gt,
                "x_pred": x_pred,
                "y_pred": y_pred,
            }
            if pred_std is not None:
                row["x_std"], row["y_std"] = pred_std[0], pred_std[1]
            for i in range(N_BINS_FULL):
                row[f"r{i}"] = float(self.last_scan[i])
            self.scan_writer.writerow(row)
            self.scan_csv_fh.flush()
        except Exception as e:
            self.get_logger().warning(f"Failed to write scan CSV row: {e}")

        # Histories for plot
        self.pred_history.append((x_pred, y_pred))
        if self.prev_time is None:
            # first frame initialization
            self.prev_time = t_stamp
            self.prev_pose = (x_true, y_true, yaw_gt)

            ekf_state = self.ekf.update(
                0.0,
                x_true, y_true,
                x_pred, y_pred,
                yaw_gt,
                0.0, 0.0
            )

        else:
            dt = max(t_stamp - self.prev_time, 1e-6)

            px0, py0, yaw0 = self.prev_pose

            dx = x_true - px0
            dy = y_true - py0

            v = math.sqrt(dx * dx + dy * dy) / dt
            omega = (yaw_gt - yaw0) / dt

            ekf_state = self.ekf.update(
                dt,
                x_true, y_true,
                x_pred, y_pred,
                yaw_gt,
                v, omega
            )

            self.prev_time = t_stamp
            self.prev_pose = (x_true, y_true, yaw_gt)

        # store EKF trajectory
        self.ekf_history.append((ekf_state[0], ekf_state[1]))
        if self.ekf_history:
           x_EKF, y_EKF = self.ekf_history[-1]
        else:
           x_EKF, y_EKF = float("nan")
        # Console log
        base_msg = (
            f"TRUE(map)({x_true:.3f},{y_true:.3f}) | "
            f"PRED({x_pred:.3f},{y_pred:.3f}) | "
            f"EKF({x_EKF:.3f},{y_EKF:.3f})|"
            f"Error_DL={e_ml:.3f}" #E_odom={e_odom:.3f}
        )
        if pred_std is not None:
            base_msg += f" STD({pred_std[0]:.3f},{pred_std[1]:.3f})"
        base_msg += f" | total inference time={total_time_ms:.2f}ms, fps={fps:.1f}"# mem={mem_mb:.1f}MB
        self.get_logger().info(base_msg)
        """
        # ==============================================
        # EKF CSV row: SLAM (map), ODOM (map), ML (map)
        # ==============================================
        try:
            if self.last_odom_in_map is not None:
                oxm, oym, oyaw_map = self.last_odom_in_map
            else:
                oxm = oym = oyaw_map = float("nan")

            self.ekf_writer.writerow([
                t_stamp,
                x_true, y_true, yaw_gt,   # SLAM GT (map)
                oxm, oym, oyaw_map,       # ODOM transformed to MAP
                x_pred, y_pred            # ML
            ])

            self.ekf_fh.flush()
        #except Exception as e:
            #self.get_logger().warning(f"Failed to write EKF CSV row: {e}")

	# ---- EKF ----
	if self.prev_time is None:
    		self.prev_time = t_stamp
    		self.prev_pose = (x_true, y_true, yaw_gt)
    		
        else:

		dt = t_stamp - self.prev_time

		px0, py0, yaw0 = self.prev_pose
		dx = x_true - px0
		dy = y_true - py0

		v = np.sqrt(dx*dx + dy*dy) / max(dt, 1e-6)
		omega = (yaw_gt - yaw0) / max(dt, 1e-6)

		ekf_state = self.ekf.update(
    				dt,
    			x_true, y_true,
    			x_pred, y_pred,
    				yaw_gt,
    				v, omega
					)

		self.ekf_history.append((ekf_state[0], ekf_state[1]))

		self.prev_time = t_stamp
		self.prev_pose = (x_true, y_true, yaw_gt)


        # ==============================================
        # Errors CSV row ( separate file)
        # ==============================================
        try:
            self.errors_writer.writerow([
                t_stamp,
                x_true, y_true, yaw_gt,
                x_pred, y_pred,
                oxm, oym, oyaw_map,
                e_ml, e_odom,
                ml_err_x, ml_err_y,
                odom_err_x, odom_err_y,
                error_ratio,
                success_flag_ml,
                success_flag_odom
            ])
            self.errors_fh.flush()
        #except Exception as e:
            #self.get_logger().warning(f"Failed to write errors CSV row: {e}")
        """
    # --------------------------
    def _update_plot(self):
        if self.plt is None:
            return

        # Odom trajectory (map frame for viz)
        odom_xs = [p[0] for p in self.odom_history]
        odom_ys = [p[1] for p in self.odom_history]

        # SLAM GT (map)
        gt_xs = [p[0] for p in self.gt_history]
        gt_ys = [p[1] for p in self.gt_history]

        # ML predictions
        pred_xs = [p[0] for p in self.pred_history]
        pred_ys = [p[1] for p in self.pred_history]
	
        # Recursive EKF
        ekf_xs = [p[0] for p in self.ekf_history]
        ekf_ys = [p[1] for p in self.ekf_history]

        self.ax.clear()
        # Drawing Map
        self.ax.plot([-0.16, 1.12], [-0.03, -0.06], 'k-', linewidth=3)
        self.ax.plot([1.12, 1.18], [-0.06, 1.155], 'k-', linewidth=3)
        self.ax.plot([1.18, -0.16], [1.16, 1.15], 'k-', linewidth=3)
        self.ax.plot([-0.16, -0.16], [1.15, -0.03], 'k-', linewidth=3)

        import matplotlib.patches as patches

        obs1 = patches.Rectangle((1.0, 0.77), 0.157, 0.10,
                             linewidth=1, edgecolor='r',
                             facecolor='gray', alpha=0.5,
                             label="Obstacle 1")
        self.ax.add_patch(obs1)

        obs2 = patches.Rectangle((0.455, 0.48), 0.150, 0.10,
                             linewidth=1, edgecolor='r',
                             facecolor='green', alpha=0.5,
                             label="Obstacle 2")
        self.ax.add_patch(obs2)

        obs3 = patches.Rectangle((-0.16, 0.75), 0.238, 0.13,
                             linewidth=1, edgecolor='r',
                             facecolor='brown', alpha=0.5,
                             label="Obstacle 3")
        self.ax.add_patch(obs3)

        if len(ekf_xs) > 0:
            self.ax.plot(ekf_xs, ekf_ys, 'b-' , linewidth =2, label="Recursive EKF")
        if len(odom_xs) > 0:
            self.ax.plot(odom_xs, odom_ys, 'k-', linewidth=1, label="Odometry")
        if len(gt_xs) > 0:
            self.ax.plot(gt_xs, gt_ys, 'r-', markersize=3, label="SLAM (Reference Pose)")
        if len(pred_xs) > 0:
            self.ax.plot(pred_xs, pred_ys, 'g.', markersize=4, label="ConvMLP Localization")

        self.ax.set_title("Live Localization: Odom vs SLAM vs ConvMLP vs Recursive EKF", fontweight = 'bold')
        self.ax.set_xlabel("x (m)", fontweight='bold')
        self.ax.set_ylabel("y (m)", fontweight='bold')
        self.ax.tick_params(axis='both', labelsize=11)
        for label in self.ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in self.ax.get_yticklabels():
            label.set_fontweight('bold')
        self.ax.axis("equal")
        self.ax.grid(True)
        legend = self.ax.legend()
        for text in legend.get_texts():
            text.set_fontweight('bold')
        self.plt.draw()
        self.plt.pause(0.001)
        
    # --------------------------
    def save_and_cleanup(self):
        # Save final plot
        if self.plt is not None:
            try:
                self._update_plot()
                png_path = os.path.join(OUT_DIR, f"inference_plot_{self.ts}.png")
                self.fig.savefig(png_path, dpi=160, bbox_inches="tight")
                self.get_logger().info(f"Saved final plot: {png_path}")
            except Exception as e:
                self.get_logger().warning(f"Failed to save final plot: {e}")

        # Close scan CSV file
        try:
            if self.scan_csv_fh:
                self.scan_csv_fh.close()
                self.get_logger().info(f"Saved scan CSV: {self.scan_csv_path}")
        except Exception:
            pass

        
        # Save summary metrics JSON (timing / memory)
        try:
            if self.total_times:
                total_arr = np.array(self.total_times)
                feat_arr = np.array(self.feature_times)
                scale_arr = np.array(self.scale_times)
                infer_arr = np.array(self.infer_times)
                mem_arr = np.array(self.mem_mb_list)

                metrics = {
                    "num_inferences": int(len(self.total_times)),
                    "total_time_ms_mean": float(total_arr.mean()),
                    "total_time_ms_std": float(total_arr.std()),
                    "total_time_ms_min": float(total_arr.min()),
                    "total_time_ms_max": float(total_arr.max()),
                    "feature_time_ms_mean": float(feat_arr.mean()),
                    "scale_time_ms_mean": float(scale_arr.mean()),
                    "infer_time_ms_mean": float(infer_arr.mean()),
                    "avg_fps": float(1000.0 / total_arr.mean()) if total_arr.mean() > 0 else None,
                    "peak_memory_mb": float(mem_arr.max()) if len(mem_arr) > 0 else None,
                }
                with open(self.metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                self.get_logger().info(f"Saved metrics JSON: {self.metrics_path}")
        except Exception as e:
            self.get_logger().warning(f"Failed to save metrics JSON: {e}")

        # Save summary error / ATE / success-rate / kidnap JSON
        try:
            if self.ml_errors:
                ml_arr = np.array(self.ml_errors, dtype=np.float64)
                od_arr = np.array(self.odom_errors, dtype=np.float64)

                ml_valid = np.isfinite(ml_arr)
                od_valid = np.isfinite(od_arr)

                def safe_stats(arr, mask):
                    if not np.any(mask):
                        return None, None, None, None
                    data = arr[mask]
                    rmse = float(np.sqrt(np.mean(data ** 2)))
                    mae = float(np.mean(np.abs(data)))
                    std = float(np.std(data))
                    maxe = float(np.max(np.abs(data)))
                    return rmse, mae, std, maxe

                ml_rmse, ml_mae, ml_std, ml_max = safe_stats(ml_arr, ml_valid)
                od_rmse, od_mae, od_std, od_max = safe_stats(od_arr, od_valid)

                n_frames = len(self.ml_errors)
                ml_success_rate = float(self.ml_success_count) / n_frames if n_frames > 0 else None
                od_success_rate = float(self.odom_success_count) / n_frames if n_frames > 0 else None

                

                summary = {
                    "position_error": {
                        "success_threshold_m": self.error_threshold,
                        "ml": {
                            "rmse_m": ml_rmse,
                            "mae_m": ml_mae,
                            "std_m": ml_std,
                            "max_error_m": ml_max,
                            "success_rate": ml_success_rate,
                        },
                    },
                }

                summary_path = os.path.join(OUT_DIR, f"summary_errors_{self.ts}.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                self.get_logger().info(f"Saved error summary JSON: {summary_path}")
        except Exception as e:
            self.get_logger().warning(f"Failed to save summary error JSON: {e}")

    # --------------------------
    def destroy_node(self):
        try:
            self.save_and_cleanup()
        except Exception as e:
            self.get_logger().warning(f"Cleanup error: {e}")
        try:
            super().destroy_node()
        except Exception:
            pass


# ---------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LocalizationInferenceLive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

