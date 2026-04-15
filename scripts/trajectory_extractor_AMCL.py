"""
Project: LiDAR Fingerprint Localization using ConvMLP pose regression
File: trajectory_extractor_AMCL.py

Description:
This script is to extract the cordinates predicted by AMCL and store it in CSV for comparison against other baselines methods. 

Author: Harsha Keladi Ganapathi
Affiliation: Robotics Lab,
             University of New Haven, CT

License: MIT License
"""


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray
import csv
import time

class AMCLMLLogger(Node):
    def __init__(self):
        super().__init__('amcl_ml_logger')

        # -------- REQUIRED FOR ROSBAG --------
        #self.declare_parameter('use_sim_time', True)

        self.amcl_pose = None
        self.ml_pose = None

        # -------- AMCL QoS (TRANSIENT LOCAL) --------
        amcl_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_cb,
            amcl_qos
        )

        # -------- ML QoS (BEST EFFORT) --------
        ml_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            Float32MultiArray,
            '/ml_pose',
            self.ml_cb,
            ml_qos
        )

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"amcl_ml_kidnap_{ts}.csv"

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'time',
            'amcl_x', 'amcl_y',
            'cov_xx', 'cov_xy', 'cov_yy',
            'ml_x', 'ml_y'
        ])

        # Timer uses ROS time now (sim time)
        self.timer = self.create_timer(0.05, self.log_row)

        self.get_logger().info(f"[OK] Logging to {self.csv_path}")

    def amcl_cb(self, msg):
        p = msg.pose.pose.position
        cov = msg.pose.covariance
        self.amcl_pose = (
            p.x, p.y,
            cov[0], cov[1], cov[7]
        )

    def ml_cb(self, msg):
        if len(msg.data) >= 2:
            self.ml_pose = (msg.data[0], msg.data[1])

    def log_row(self):
        # Log even if ML temporarily missing (important for kidnap)
        if self.amcl_pose is None:
            return

        row = [
            self.get_clock().now().nanoseconds * 1e-9,
            *self.amcl_pose,
        ]

        if self.ml_pose is not None:
            row += list(self.ml_pose)
        else:
            row += [None, None]

        self.writer.writerow(row)
        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = AMCLMLLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

