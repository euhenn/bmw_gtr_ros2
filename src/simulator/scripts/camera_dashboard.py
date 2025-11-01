#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import subprocess
import signal
import os

class CameraDashboard(Node):
    def __init__(self):
        super().__init__('camera_dashboard')

        # Dictionary of topics with window names
        self.topics = {
            'Front Camera': '/automobile/camera/image_raw',        # 320x240
            'Top Camera': '/automobile/camera_top/image_raw',      # 640x480
            'Overhead Camera': '/sim/overhead_camera/image_raw'    # 1068x708
        }

        self.bridge = CvBridge()
        self.images = {name: None for name in self.topics}
        self.running = True

        # Create subscribers for each camera
        for name, topic in self.topics.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, n=name: self.image_callback(msg, n),
                10
            )
            self.get_logger().info(f"Subscribed to {topic}")

        # Create windows for each camera
        for window_name in self.topics.keys():
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("Camera Dashboard started. Press 'q' in any window to close all and kill gzserver.")

    def image_callback(self, msg, name):
        # Convert ROS Image to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.images[name] = cv_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert image for {name}: {e}")

    def update_display(self):
        # Display each image in its own window
        for name, img in self.images.items():
            if img is not None:
                # Display the image without text overlay
                cv2.imshow(name, img)
            else:
                # Show black image if no image received
                black_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow(name, black_img)

        # Check for key press - if 'q' is pressed in any window, return False
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        return True

    def kill_gzserver(self):
        """Kill gzserver processes"""
        self.get_logger().info("Killing gzserver processes...")
        try:
            # Kill gzserver processes
            subprocess.run(['pkill', '-f', 'gzserver'], check=False)
            # Also kill gazebo processes if any
            subprocess.run(['pkill', '-f', 'gazebo'], check=False)
            self.get_logger().info("gzserver processes killed successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to kill gzserver: {e}")

    def destroy_windows(self):
        """Close all OpenCV windows and kill gzserver"""
        cv2.destroyAllWindows()
        # Additional cleanup to ensure all windows are closed
        for i in range(1, 10):
            cv2.waitKey(1)
        
        # Kill gzserver when windows are closed
        self.kill_gzserver()


def main(args=None):
    rclpy.init(args=args)
    node = CameraDashboard()
    
    try:
        # Main loop
        while rclpy.ok():
            # Update displays
            if not node.update_display():
                node.get_logger().info("Quit signal received. Shutting down...")
                break
                
            # Small sleep to prevent excessive CPU usage
            rclpy.spin_once(node, timeout_sec=0.01)
            
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        # Cleanup - this will also kill gzserver
        node.destroy_windows()
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("Camera Dashboard shutdown complete.")


if __name__ == '__main__':
    main()