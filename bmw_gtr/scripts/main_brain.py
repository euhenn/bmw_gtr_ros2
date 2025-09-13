#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from automobile_data_simulator import AutomobileDataSimulator
from mpc_kin import MPC_KinematicBicycle
import numpy as np
import time
import threading


class CarControllerNode(Node):
    def __init__(self):
        super().__init__('car_controller_node')

        # Init simulator
        self.car = AutomobileDataSimulator(
            trig_control=True,
            trig_bno=True,
            trig_enc=True,
            trig_gps=True
        )

        # Control step (seconds)
        self.dt = 0.1   # MPC update frequency (2 Hz) # THESE TWO MUST BE THE SAME
        # MPC setup
        self.mpc = MPC_KinematicBicycle(dt_ocp=self.dt, N_horizon=10)

        # Trajectory index
        self.idx = 0
        self.v_cmd = 0.0

        # Background simulator spin thread
        self._sim_running = True
        self.sim_thread = threading.Thread(target=self.spin_simulator, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("Car Controller Node started with fixed-step MPC and background simulator.")

    def spin_simulator(self):
        """Continuously spin the simulator at high rate (50 Hz)."""
        rate = 0.02  # 20 ms = 50 Hz
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            time.sleep(rate)

    def get_current_state(self):
        """Get current vehicle state from simulator."""
        x = float(self.car.x_true)
        y = float(self.car.y_true)
        yaw = np.deg2rad(float(self.car.yaw_true))
        v = float(self.car.filtered_encoder_velocity)
        return np.array([x, y, yaw, self.v_cmd])

    def apply_control(self, v, delta):
        """Send control commands to simulator (or Gazebo if mapped)."""
        self.car.drive_speed(v)
        self.car.drive_angle(np.rad2deg(delta))

    def find_closest_index(self, x, y):
        """Find the closest trajectory index to the current position."""
        traj = self.mpc.trajectory
        diffs = traj[:2, :].T - np.array([x, y])  # shape (N,2)
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    def run(self):
        """Main MPC control loop with look-ahead indexing."""
        next_time = time.time()
        lookahead_steps = 5   # number of steps ahead in trajectory (~0.5s if dt=0.1)
        
        while rclpy.ok():
            state = self.get_current_state()
            x, y, yaw, _ = state

            # --- Compute trajectory index dynamically ---
            closest_idx = self.find_closest_index(x, y)
            self.idx = min(
                closest_idx + lookahead_steps,
                self.mpc.trajectory.shape[1] - self.mpc.N_horizon - 1
            )

            # Check stop condition
            if x > 10 and y < 0:
                self.get_logger().info("Stop condition reached (x>10, y<0).")
                self.apply_control(0.0, 0.0)
                break

            # Get reference horizon
            traj_horizon = self.mpc.get_reference_segment(self.idx)

            # Solve MPC
            t0 = time.time()
            self.v_cmd, delta_cmd = self.mpc.solve(state[:3], traj_horizon)
            self.get_logger().info(f"MPC solve took {time.time()-t0:.3f}s")

            # Apply control
            self.apply_control(self.v_cmd, delta_cmd)

            # Log
            self.get_logger().info(
                f"Step {self.idx} | Pos: ({x:.2f}, {y:.2f}) "
                f"Yaw: {np.rad2deg(yaw):.2f}° | Vel: {state[3]:.2f} m/s | "
                f"Cmd -> v: {self.v_cmd:.2f} m/s, δ: {np.rad2deg(delta_cmd):.1f}°"
            )

            # --- Real-time wait ---
            next_time += self.dt
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.get_logger().warn(f"Loop overran by {-sleep_time:.3f}s")


    def shutdown(self):
        """Stop everything cleanly."""
        self._sim_running = False
        self.sim_thread.join(timeout=1.0)
        self.apply_control(0.0, 0.0)


def main():
    rclpy.init()
    node = CarControllerNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
