#!/usr/bin/python3
import rclpy
from automobile_data_simulator import AutomobileDataSimulator
import time
import math
import sys
import select
import tty
import termios
import os

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
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.speed_increment = 0.1
        self.steering_increment = 5.0  # degrees
        self.max_speed = 2.0
        self.max_steering = 30.0
        
        # Time management
        self.control_interval = 0.1  # 10 Hz (0.1 seconds)
        self.last_control_time = time.time()
        
        # Set a timer to run the control loop
        self.timer = self.create_timer(self.control_interval, self.control_loop)
        
        # Save original terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # Set terminal to raw mode for non-blocking input
        tty.setraw(sys.stdin.fileno())
        
        # Error tracking
        self.timeout_error_occurred = False
    
    def get_key(self):
        """Check if a key is pressed and return it"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    
    def control_loop(self):
        """Main control loop that processes keyboard input and applies controls"""
        # Check if we've already had a timeout error
        if self.timeout_error_occurred:
            return
            
        # Measure start time
        start_time = time.time()
        
        # Process input and apply controls
        key = self.get_key()
        
        if key:
            self.process_key_input(key)
        
        # Apply current controls
        self.drive_speed(self.current_speed)
        self.drive_angle(self.current_steering)
        
        # Clear terminal and show live status
        print("\033c")   # ANSI clear screen
        print("Q: Quit\n")
        print(f"Speed: {self.current_speed:.1f} m/s")
        print(f"Steering: {self.current_steering:.1f}°")
        print(f"X: {self.x_true:.1f} m")
        print(f"Y: {self.y_true:.1f} m")
        print(f"Yaw: {self.yaw_true:.1f}°")
        
        # Check if we exceeded the time limit
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display execution time for monitoring
        print(f"Loop time: {execution_time:.6f} s")
        
        if execution_time > self.control_interval:
            #self.timeout_error_occurred = True
            print(f"ERROR: Control loop exceeded time limit of {self.control_interval}s")
            print(f"Actual execution time: {execution_time:.6f}s")
            #print("Stopping simulation...")
            #self.cleanup()
            #rclpy.shutdown()
            #sys.exit(1)

    def process_key_input(self, key):
        """Process keyboard input and update control parameters"""
        if key == 'w' or key == 'W':
            # Increase speed
            self.current_speed = min(self.current_speed + self.speed_increment, self.max_speed)
        
        elif key == 's' or key == 'S':
            # Decrease speed
            self.current_speed = max(self.current_speed - self.speed_increment, -self.max_speed)
        
        elif key == 'a' or key == 'A':
            # Steer left
            self.current_steering = max(self.current_steering - self.steering_increment, -self.max_steering)
        
        elif key == 'd' or key == 'D':
            # Steer right
            self.current_steering = min(self.current_steering + self.steering_increment, self.max_steering)
        
        elif key == ' ':
            # Spacebar - stop and center steering
            self.current_speed = 0.0
            self.current_steering = 0.0
        
        elif key == 'q' or key == 'Q':
            # Quit
            self.cleanup()
            rclpy.shutdown()
            sys.exit(0)
    
    def stop(self):
        """Stop the car completely"""
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.drive_speed(0.0)
        self.drive_angle(0.0)
    
    def cleanup(self):
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.stop()

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
        # Clean up and stop the car
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()