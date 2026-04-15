"""
Project: LiDAR Fingerprint Localization using ConvMLP pose regression
File: teleop_with_trigger.py

Description: This scripts, provide a keyboard teleoperation, to move the robot while collecting the data and while testing the model in inference process. check below about keyborad buttons and their operation.     

Author: Harsha Keladi Ganapathi
Affiliation: Robotics Lab,
             University of New Haven, CT

License: MIT License
"""


#!/usr/bin/env python3
"""
teleop_auto_trigger.py

- Keyboard teleop for TurtleBot3
- w = forward
- x = reverse
- a = rotate left
- d = rotate right
- s = stop
- r = auto-rotate
- Automatically publishes /capture_trigger while moving
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import sys, termios, tty, threading, time
import math

MOVE_BINDINGS = {
    'w': (0.2, 0.0),     # forward
    'x': (-0.2, 0.0),    # reverse
    'a': (0.0, 0.2),     # turn left
    'd': (0.0, -0.2),    # turn right
    's': (0.0, 0.0),     # STOP
}

class TeleopAutoTrigger(Node):
    def __init__(self):
        super().__init__('teleop_auto_trigger')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.trigger_pub = self.create_publisher(Bool, '/capture_trigger', 10)

        self.settings = termios.tcgetattr(sys.stdin)
        self.running = True
        self.busy_rotating = False

        self.current_cmd = Twist()
        self.trigger_count = 0

        self.get_logger().info("\nTeleop Started:")
        self.get_logger().info("Controls: w/x = forward/back, a/d = left/right, s = stop, CTRL+C to quit\n")

        threading.Thread(target=self.keyboard_loop, daemon=True).start()
        threading.Thread(target=self.trigger_loop, daemon=True).start()

    def auto_rotate(self, step_deg=10, speed=0.4):
        if self.busy_rotating:
            return

        self.busy_rotating = True
        self.get_logger().info("Starting auto-rotation capture...")

        # Stop robot first
        stop = Twist()
        self.cmd_pub.publish(stop)
        time.sleep(0.5)

        step_rad = math.radians(step_deg)
        rotation_time = step_rad / speed  # t = theta / w

        for angle in range(0, 360, step_deg):

            # Rotate by step_deg
            twist = Twist()
            twist.angular.z = speed
            self.cmd_pub.publish(twist)
            time.sleep(rotation_time)

            # Stop after each step
            self.cmd_pub.publish(stop)
            time.sleep(0.1)

            # Trigger data capture
            self.trigger_pub.publish(Bool(data=True))
            self.trigger_count += 1
            print(f"\rAuto Triggers: {self.trigger_count}", end="")

        # Final stop
        self.cmd_pub.publish(stop)
        self.get_logger().info("\nAuto-rotation complete.")
        self.busy_rotating = False
    
    def keyboard_loop(self):
        while self.running:
            key = self.get_key()
            
            if key == 'r':
                threading.Thread(target=self.auto_rotate, daemon=True).start()
                continue

            if key in MOVE_BINDINGS and not self.busy_rotating:
                lin, ang = MOVE_BINDINGS[key]
                self.current_cmd.linear.x = lin
                self.current_cmd.angular.z = ang
                self.cmd_pub.publish(self.current_cmd)

            elif key == '\x03':  # Ctrl+C
                self.running = False
                break

        stop = Twist()
        self.cmd_pub.publish(stop)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        rclpy.shutdown()

    def trigger_loop(self):
        while self.running:
            moving = abs(self.current_cmd.linear.x) > 0.001 or abs(self.current_cmd.angular.z) > 0.001

            if moving:
                self.trigger_pub.publish(Bool(data=True))
                self.trigger_count += 1
                print(f"\rTriggers: {self.trigger_count}", end="")

            time.sleep(0.15)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key


def main(args=None):
    rclpy.init(args=args)
    node = TeleopAutoTrigger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()

