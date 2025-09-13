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

        # MPC setup
        self.mpc = MPC_KinematicBicycle(dt_ocp=0.2, N_horizon=20)

        # Control step (seconds)
        self.dt = 0.5   # MPC update frequency (2 Hz)

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

    def run(self):
        """Main MPC control loop (runs slower than simulator spin)."""
        next_time = time.time()
        while rclpy.ok():
            start_time = time.time()

            # Current state (already updated in background)
            state = self.get_current_state()

            # Check if trajectory finished
            if self.idx >= self.mpc.trajectory.shape[1] - self.mpc.N_horizon - 1:
                self.get_logger().info("End of trajectory reached")
                self.apply_control(0.0, 0.0)
                break

            # Get reference segment for horizon
            traj_horizon = self.mpc.get_reference_segment(self.idx)

            # Solve MPC
            t0 = time.time()
            self.v_cmd, delta_cmd = self.mpc.solve(state[:3], traj_horizon)
            self.get_logger().info(f"MPC solve took {time.time()-t0:.3f}s")

            # Apply control
            self.apply_control(self.v_cmd, delta_cmd)

            # Logging
            self.get_logger().info(
                f"Step {self.idx} | Pos: ({state[0]:.2f}, {state[1]:.2f}) "
                f"Yaw: {np.rad2deg(state[2]):.2f}° | Vel: {state[3]:.2f} m/s | "
                f"Cmd -> v: {self.v_cmd:.2f} m/s, δ: {np.rad2deg(delta_cmd):.1f}°"
            )

            self.idx += 1

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
