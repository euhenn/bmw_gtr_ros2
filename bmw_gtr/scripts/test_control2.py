#!/usr/bin/python3
import rclpy
from automobile_data_simulator import AutomobileDataSimulator
import time
import sys


class CarControllerNode(AutomobileDataSimulator):
    def __init__(self):
        # Initialize the AutomobileDataSimulator with desired features
        super().__init__(
            trig_control=True,
            trig_bno=True,
            trig_enc=True,
            trig_gps=True
        )
        
        # Control parameters
        self.current_speed = 0.5       # start speed [m/s]
        self.current_steering = 0.0    # start steering [°]
        self.max_speed = 2.0
        self.max_steering = 30.0

        # Fixed-step loop configuration
        self.control_interval = 0.1  # 10 Hz
        self.timer = self.create_timer(self.control_interval, self.control_loop)

        # Error tracking
        self.timeout_error_occurred = False

    def control_loop(self):
        """Main control loop running at fixed time steps"""
        if self.timeout_error_occurred:
            return

        start_time = time.time()

        # Example: simple oscillation of steering to show it works
        self.current_steering = 15.0 * \
            (1 if int(time.time()) % 4 < 2 else -1)

        # Keep speed constant
        self.current_speed = 1.0

        # Apply current controls
        self.drive_speed(self.current_speed)
        self.drive_angle(self.current_steering)

        # Clear terminal and show live status
        print("\033c")   # ANSI clear screen
        print(f"Speed: {self.current_speed:.1f} m/s")
        print(f"Steering: {self.current_steering:.1f}°")
        print(f"X: {self.x_true:.1f} m")
        print(f"Y: {self.y_true:.1f} m")
        print(f"Yaw: {self.yaw_true:.1f}°")

        # Check loop execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Loop time: {execution_time:.6f} s")

        if execution_time > self.control_interval:
            self.timeout_error_occurred = True
            print(f"ERROR: Control loop exceeded {self.control_interval}s")
            print("Stopping simulation...")
            self.cleanup()
            rclpy.shutdown()
            sys.exit(1)


    def cleanup(self):
        """Clean shutdown"""
        self.stop() # Stop the car


def main(args=None):
    rclpy.init(args=args)
    node = CarControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
