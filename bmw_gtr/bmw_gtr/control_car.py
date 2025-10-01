#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
import json
import sys, termios, tty, select

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class KeyboardCarController(Node):
    def __init__(self):
        super().__init__('keyboard_car_controller')

        # Publisher to car command topic
        self.drive_pub = self.create_publisher(String, '/automobile/command', 10)

        # Subscriber to camera topic
        self.bridge = CvBridge()
        self.frame = None
        self.create_subscription(Image, '/automobile/camera_top/image_raw', self.camera_callback, 10)

        # Control state
        self.speed = 0.0
        self.steering = 0.0

        # Timer for publishing commands
        self.create_timer(0.1, self.publish_command)  # 10 Hz

        cv.namedWindow("camera", cv.WINDOW_NORMAL)
        cv.resizeWindow("camera", 640, 480)

        self.get_logger().info("KeyboardCarController started. Use WSAD to drive, Q to quit.")

    def camera_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv.imshow("camera", self.frame)
            cv.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def publish_command(self):
        # Send current speed and steering
        msg_speed = String()
        msg_steer = String()
        msg_speed.data = json.dumps({'action': '1', 'speed': float(self.speed)})
        msg_steer.data = json.dumps({'action': '2', 'steerAngle': float(np.rad2deg(self.steering))})
        self.drive_pub.publish(msg_speed)
        self.drive_pub.publish(msg_steer)

    def update_keyboard(self):
        key = get_key_nonblocking()
        if key:
            if key == 'w':   # accelerate
                self.speed += 0.1
            elif key == 's':  # brake/reverse
                self.speed -= 0.1
            elif key == 'a':  # steer left
                self.steering -= 0.05  # rad
            elif key == 'd':  # steer right
                self.steering += 0.05  # rad
            elif key == ' ':  # space = stop
                self.speed = 0.0
                self.steering = 0.0
            elif key == 'q':  # quit
                self.get_logger().info("Exiting...")
                return False

            self.get_logger().info(f"Speed={self.speed:.2f} m/s, Steer={np.rad2deg(self.steering):.1f} deg")
        return True


# --- helper for non-blocking keyboard input ---
def get_key_nonblocking():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def main(args=None):
    rclpy.init(args=args)

    settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    node = KeyboardCarController()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if not node.update_keyboard():
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
