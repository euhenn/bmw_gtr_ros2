#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from automobile_data_simulator import AutomobileDataSimulator
from time import sleep, time
import sys, termios, tty, select
import cv2

TARGET_FPS = 30.0  # Target frames per second for main loop


def get_key(timeout=0.01):
    """Non-blocking single key reader"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        else:
            return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class AutomobileDataDisplay(AutomobileDataSimulator):
    """
    ROS2 node that displays automobile data in the terminal,
    with keyboard control for speed and steering.
    """

    def __init__(self):
        super().__init__(
            trig_control=True,  # enable control publishers
            trig_bno=True,
            trig_enc=True,
            trig_gps=True,
            trig_cam=False
        )

        self.last_display_time = 0.0
        self.speed_cmd = 0.0      # m/s
        self.steer_cmd = 0.0      # degrees
        self.drive_dir = 1        # +1 forward, -1 backward
        self.running = True

        print("🚗 Automobile Data Display + Keyboard Control Started")
        print("Controls:")
        print("  i/k - increase/decrease speed (0–2 m/s, step 0.2)")
        print("  a/d - decrease/increase steering (-30°–+30°, step 5°)")
        print("  w/s - forward/reverse direction")
        print("  space - stop (0 m/s, 0°)")
        print("  q or Ctrl+C - exit\n")

    def display_data(self):
        """Clear terminal and print current data values."""
        print("\033c", end="")  # Clear screen
        print("=== 🚗 Automobile Data Display ===\n")
        print(f"Encoder Distance: {self.encoder_distance:.3f} m")
#       print(f"Sensors:          x_gps={self.x_gps:.3f}, y_gps={self.y_gps:.3f}, yaw_imu={self.yaw_true:.2f} rad")
        print(f"localisation:      x_est={self.x_est:.3f}, y_est={self.y_est:.3f}") #, yaw_est={self.yaw_est:.2f} rad")
        print(f"Steering Angle:   {self.curr_steer:.3f} deg")
        print(f"Speed:            {self.speed:.3f} m/s")
        print(f"Sim time IMU:     {self.timestamp:.3f} s")
        print("\n---------------------------------------")
        print(f"Commanded Speed:  {self.speed_cmd * self.drive_dir:.2f} m/s")
        print(f"Commanded Angle:  {self.steer_cmd:.1f}°")
        print("\n[i]↑ speed | [k]↓ speed | [w/s] direction | [a/d] steering | [space] stop | [q] quit")

    def display_camera(self):
        """Show the latest camera frame if available."""
        if hasattr(self, "frame") and self.frame is not None:
            cv2.imshow("Front camera", self.frame)
            # Needed to refresh image window and check for window close
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def process_key(self, key):
        """Handle keyboard input"""
        if key is None:
            return

        if key.lower() == 'i':
            self.speed_cmd = min(self.speed_cmd + 0.2, 2.0)
        elif key.lower() == 'k':
            self.speed_cmd = max(self.speed_cmd - 0.2, 0.0)
        elif key.lower() == 'a':
            self.steer_cmd = max(self.steer_cmd - 5.0, -30.0)
        elif key.lower() == 'd':
            self.steer_cmd = min(self.steer_cmd + 5.0, 30.0)
        elif key.lower() == 'w':
            self.drive_dir = 1
        elif key.lower() == 's':
            self.drive_dir = -1
        elif key == ' ':
            self.speed_cmd = 0.0
            self.steer_cmd = 0.0
            self.stop()
            return
        elif key.lower() == 'q':
            self.running = False
            return
        else:
            return

        # Send commands
        self.drive_speed(self.speed_cmd * self.drive_dir)
        self.drive_angle(self.steer_cmd)

    def run(self):
        """Main loop with keyboard control"""
        try:
            while rclpy.ok() and self.running:
                rclpy.spin_once(self, timeout_sec=0.01)
                
                loop_start_time = time()
                #### START - Loop actions ####

                key = get_key(timeout=0.01)
                self.process_key(key)
                self.display_data()
                #self.display_camera()

                # Optional: limit loop based on IMU sensor update rate; self.timestamp is coming from IMU topic
                #if self.timestamp - self.last_display_time > 3.0: # e.g., every 3 seconds in the simulation, not real time
                #    self.last_display_time = self.timestamp
                #    self.display_data()
                #else:
                #    sleep(3 - (self.timestamp - self.last_display_time))

                #### END - Loop actions ####
                loop_end_time = time()

                loop_duration = loop_end_time - loop_start_time
                if loop_duration < 1 / TARGET_FPS:
                    sleep(1 / TARGET_FPS - loop_duration)
                else:
                    print("Warning: Loop is running slower than target FPS.")

        except KeyboardInterrupt:
            print("\n Ctrl+C pressed — shutting down.")
        finally:
            self.stop()
            print("Destroying node...")
            self.destroy_node()
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = AutomobileDataDisplay()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
