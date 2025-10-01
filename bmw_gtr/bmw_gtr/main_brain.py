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
        self.dt = 0.05
        # MPC setup
        self.mpc = MPC_KinematicBicycle(dt_ocp=self.dt, N_horizon=40)

        # Align trajectory yaw with current simulator yaw
        self.align_trajectory_with_vehicle()

        # Trajectory index
        self.idx = 0
        self.v_cmd = 0.0

        # Background simulator spin thread
        self._sim_running = True
        self.sim_thread = threading.Thread(target=self.spin_simulator, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("Car Controller Node started with fixed-step MPC and trajectory aligned.")

    def align_trajectory_with_vehicle(self):
        """Rotate trajectory yaw to match simulator initial yaw."""
        x0 = float(self.car.x_true)
        y0 = float(self.car.y_true)
        yaw0 = np.deg2rad(float(self.car.yaw_true))

        traj = self.mpc.trajectory.copy()
        traj_yaw0 = traj[2, 0]

        yaw_offset = yaw0 - traj_yaw0
        traj[2, :] += yaw_offset  # align entire trajectory yaw

        # Optionally wrap yaw to [-pi, pi]
        #traj[2, :] = (traj[2, :] + np.pi) % (2 * np.pi) - np.pi

        self.mpc.trajectory = traj
        self.get_logger().info(f"Trajectory yaw aligned with vehicle: offset={np.rad2deg(yaw_offset):.2f}°")

    def spin_simulator(self):
        rate = 0.02
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            time.sleep(rate)

    def get_current_state(self):
        x = float(self.car.x_true)
        y = float(self.car.y_true)
        yaw = np.deg2rad(float(self.car.yaw_true))
        v = float(self.car.filtered_encoder_velocity)
        return np.array([x, y, yaw, self.v_cmd])

    def apply_control(self, v, delta):
        self.car.drive_speed(v)
        self.car.drive_angle(np.rad2deg(delta))

    def find_closest_index(self, x, y):
        traj = self.mpc.trajectory
        diffs = traj[:2, :].T - np.array([x, y])
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    def run(self):
        next_time = time.time()
        lookahead_steps = 10
        while rclpy.ok():
            state = self.get_current_state()
            x, y, yaw, _ = state

            closest_idx = self.find_closest_index(x, y)
            self.idx = min(
                closest_idx + lookahead_steps,
                self.mpc.trajectory.shape[1] - self.mpc.N_horizon - 1
            )

            if x > 4 and y < 0:
                self.get_logger().info("Stop condition reached")
                self.apply_control(0.0, 0.0)
                break

            traj_horizon = self.mpc.get_reference_segment(self.idx)

            t0 = time.time()
            self.v_cmd, delta_cmd = self.mpc.solve(state[:3], traj_horizon)
            self.get_logger().info(f"MPC solve took {time.time()-t0:.3f}s")

            self.apply_control(self.v_cmd, delta_cmd)

            self.get_logger().info(
                f"Step {self.idx} | Pos: ({x:.2f}, {y:.2f}) "
                f"Yaw: {np.rad2deg(yaw):.2f}° | Vel: {state[3]:.2f} m/s | "
                f"Cmd -> v: {self.v_cmd:.2f} m/s, δ: {np.rad2deg(delta_cmd):.1f}°"
            )

            next_time += self.dt
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.get_logger().warn(f"Loop overran by {-sleep_time:.3f}s")

    def shutdown(self):
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
