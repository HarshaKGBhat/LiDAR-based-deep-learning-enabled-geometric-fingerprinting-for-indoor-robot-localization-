"""
Project: LiDAR Fingerprint Localization using ConvMLP pose regression
File: fingerprint_collector.py

Description:
This script, collects the hybrid fetaures as the robot traverse and store it in the csv format (Note that reference pose is required to store as a target value so SLAM need to be run simultaneously with this script, for more details look into (Commands to run pipeline readme file). 

Author: Harsha Keladi Ganapathi
Affiliation: Robotics Lab,
             University of New Haven, CT

License: MIT License
"""


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import csv
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, entropy
from numpy.fft import rfft

# ============================================================
# PARAMETERS
# ============================================================
N_BINS_FULL = 360   

def interpolate_to_360(scan):
    n = len(scan)
    if n == 360:
        return np.array(scan, dtype=float)

    # Original angles
    old_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Virtual angles (360)
    new_angles = np.linspace(0, 2 * np.pi, N_BINS_FULL, endpoint=False)

    return np.interp(new_angles, old_angles, scan)

def extract_optimized_features(scan, max_range):
    d = np.array(scan, dtype=np.float64)

    # Ensure 360 bins
    if len(d) != N_BINS_FULL:
        d = interpolate_to_360(d)

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


# ============================================================
# TF quaternion → yaw
# ============================================================
def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    return math.atan2(siny, cosy)


# ============================================================
# MAIN CLASS
# ============================================================
class FingerprintCollectorLiveCartographer(Node):

    def __init__(self):
        super().__init__("fingerprint_collector_real")

        # -----------------------------
        # TIME-STAMPED FILES
        # -----------------------------
        t = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.csv_filename = f"lidar_fingerprint_{t}.csv"
        self.json_filename = f"samples_{t}.json"
        self.plot_filename = f"trajectory_{t}.png"

        self.csv_file = open(self.csv_filename, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        feat_keys = [
            "std_dist", "min_dist", "max_dist", "skew", "kurtosis", "roughness",
            "jump_rate", "entropy_dist", "fft_std", "aspect_ratio",
            "num_clusters", "clearance_proxy"
        ]

        header = ["id", "x", "y", "yaw", "sin_yaw", "cos_yaw"] + \
                 feat_keys + [f"r{i}" for i in range(N_BINS_FULL)]

        self.writer.writerow(header)
        self.get_logger().info(f"Saving CSV to {self.csv_filename}")

        self.samples = []
        self.sample_id = 0

        # TF LISTENER
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # LIDAR SUBSCRIBER
        scan_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.latest_scan = None
        self.range_max = 3.5

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile=scan_qos
        )

        # -----------------------------
        # LIVE TRAJECTORY PLOT
        # -----------------------------
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_title("Robot Trajectory (live)")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(True)

        self.traj_x = []
        self.traj_y = []

        self.traj_plot, = self.ax.plot([], [], "-b", linewidth=1)
        self.current_point, = self.ax.plot([], [], "or", markersize=4)

        self.get_logger().info("Collector ready: receiving /scan and TF.")

    # =======================================================
    # LASER CALLBACK
    # =======================================================
    def scan_callback(self, msg):
        scan = np.array(msg.ranges, dtype=float)
        scan = np.nan_to_num(scan, nan=msg.range_max)

        # Interpolate scan to 360 bins
        scan_360 = interpolate_to_360(scan)

        self.range_max = msg.range_max
        self.latest_scan = scan_360

        # Try to get TF pose
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            yaw = quat_to_yaw(tf.transform.rotation)
        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return

        # Update live trajectory
        self.traj_x.append(x)
        self.traj_y.append(y)

        # Compute features
        feats = extract_optimized_features(scan_360, self.range_max)

        # Save CSV
        self.sample_id += 1
        row = [
            self.sample_id,
            x, y, yaw,
            math.sin(yaw), math.cos(yaw),
        ] + [feats[k] for k in feats.keys()] + list(scan_360)

        self.writer.writerow(row)
        self.csv_file.flush()

        # Save JSON entry
        self.samples.append({
            "id": self.sample_id,
            "x": x, "y": y, "yaw": yaw,
            "sin_yaw": math.sin(yaw),
            "cos_yaw": math.cos(yaw),
            "features": feats,
            "ranges": list(scan_360)
        })

        # Live plot update
        self.traj_plot.set_data(self.traj_x, self.traj_y)
        self.current_point.set_data([x], [y])

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.get_logger().info(f"Sample {self.sample_id} saved.")

    # =======================================================
    # CLEANUP
    # =======================================================
    def destroy_node(self):
        with open(self.json_filename, "w") as jf:
            json.dump(self.samples, jf, indent=2)
        self.get_logger().info(f"Saved JSON: {self.json_filename}")

        self.fig.savefig(self.plot_filename)
        self.get_logger().info(f"Saved trajectory PNG: {self.plot_filename}")

        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FingerprintCollectorLiveCartographer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

