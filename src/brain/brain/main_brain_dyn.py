#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

from automobile_data_simulator import AutomobileDataSimulator
from mpc_controller_dyn import MPC_DynamicBicycle

# -------------------------------
# Control loop parameters
# -------------------------------
TARGET_FPS = 50.0         # target loop frequency [Hz]
DT = 1.0 / TARGET_FPS     # fixed timestep for velocity integration [s]


class CarControllerNode(Node):
    def __init__(self):
        super().__init__('car_controller_node')

        self.loop_durations = []

        # --- Logging setup ---
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"mpc_log_{timestamp}.csv")
        self.log_file = open(self.log_path, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "t", "idx",
            "x", "y", "yaw", "vx", "vy","omega", "delta_actual",
            "e_psi", "e_y",
            "a_cmd", "delta_cmd",
            "v_cmd"
        ])

        # --- Initialize simulator ---
        self.car = AutomobileDataSimulator(
            trig_control=True,
            trig_bno=True,
            trig_enc=True,
            trig_gps=True
        )
        
        self.prev_yaw = float(self.car.yaw_true)
        self.first_cycle = True

        # --- Background simulator thread ---
        self._sim_running = True
        self.sim_thread = threading.Thread(target=self.spin_simulator, daemon=True)
        self.sim_thread.start()
        self.car.drive_speed(0.0)

        # --- Initialize MPC ---
        self.mpc = MPC_DynamicBicycle(ds=0.01, N_horizon=100)

        # --- Initialize control variables ---
        self.get_logger().info(f"Control loop target rate: {TARGET_FPS} Hz")


    # ---------------------------------------------------------------
    # SIMULATION THREAD
    # ---------------------------------------------------------------
    def spin_simulator(self):
        """Continuously update simulator data in background."""
        rate = 0.01  # 50 Hz simulator update
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            time.sleep(rate)

    # ---------------------------------------------------------------
    # STATE ACCESSORS
    # ---------------------------------------------------------------
    def get_current_state(self):
        """Return [x, y, yaw, v]."""

        x_gps = float(self.car.x_est)
        y_gps = float(self.car.y_est)
        yaw_imu = float(self.car.yaw_true)
        vx = float(self.car.filtered_encoder_velocity)

        omega = (yaw_imu - self.prev_yaw) / DT
        self.prev_yaw = yaw_imu

        # Estimate lateral velocity (vy)
        delta = np.deg2rad(self.car.curr_steer)
        beta = np.arctan((self.mpc.lr / self.mpc.L) * np.tan(delta))
        vy = vx * np.tan(beta)

        return np.array([ x_gps, y_gps, yaw_imu, vx, vy, omega])

    def apply_control(self, v, delta):
        """Send control commands to simulator."""
        self.car.pub_speed(v)
        self.car.drive_angle(np.rad2deg(delta))

    # ---------------------------------------------------------------
    # MAIN CONTROL LOOP (WITH FPS LIMITER)
    # ---------------------------------------------------------------
    
    def run(self):
        self.get_logger().info("Starting main control loop...")
        a_prev, delta_prev = 0.0, 0.0
        delay_sec = 0.17   # measured actuation delay
        loop_duration = 0


        while rclpy.ok():
            loop_start = time.time()

            if(self.first_cycle):
                self.first_cycle = False
                _=self.get_current_state()
                continue


            try:
                # === 1. Get current state ===
                x, y, yaw, vx, vy, omega = self.get_current_state()

                if (False):
                    # === 2. Predict future state to compensate for delay ===
                    # Simple kinematic bicycle forward propagation for delay_sec seconds
                    L = self.mpc.L
                    beta = np.arctan(self.mpc.lr * np.tan(delta_prev) / self.mpc.L)
                    x_pred = x + v * np.cos(yaw + beta) * delay_sec
                    y_pred = y + v * np.sin(yaw + beta) * delay_sec
                    yaw_pred = yaw + (v / L) * np.sin(beta) / np.cos(beta) * delay_sec
                    v_pred = v + a_prev * delay_sec
                    if v_pred < 0.0:
                        v_pred = 0.0
                    elif v < 0.05 and a_prev > 0.0:
                        v_pred = max(v_pred, 0.05)
                    state_ocp, idx = self.mpc.get_state(x_pred, y_pred, yaw_pred, v_pred)
                else:
                    # === 2. Transform to spatial coordinates ===#
                    state_ocp, idx = self.mpc.get_state(x, y, yaw, vx, vy, omega)

                # === 3. Use the spatial coordinates ===
                e_psi, e_y = state_ocp[0], state_ocp[1]

                # === 4. Solve MPC ===
                a_cmd, delta_cmd = self.mpc.solve(state_ocp, idx,
                                                  warm_start=None)#np.array([a_prev, delta_prev]))

                # === 5. Integrate control & apply ===

                v_cmd = max(0.01, vx + (a_cmd * loop_duration))
                self.apply_control(v_cmd, delta_cmd)

                a_prev, delta_prev = a_cmd, delta_cmd

            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")
                self.apply_control(0.0, 0.0)
                break

            # === 6. Logging ===
            self.log_writer.writerow(map(float, [
                time.time(),
                idx,
                x, y, yaw, vx, vy, omega, self.car.curr_steer,
                e_psi, e_y,
                a_cmd, delta_cmd,
                v_cmd
            ]))

            # === 7. FPS limiter ===
            loop_duration = time.time() - loop_start
            self.loop_durations.append(loop_duration)
            target_period = 1.0 / TARGET_FPS # 0.066666 secods
            if loop_duration < target_period: 
                time.sleep(target_period - loop_duration)
                #self.get_logger().warn(
                #    f"        Loop: took {loop_duration:.3f}s"
                #)
            else:
               self.get_logger().warn(
                   f"Loop overrun: took {loop_duration:.3f}s > {target_period:.3f}s"
               )

    # ---------------------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------------------
    def shutdown(self):
        self._sim_running = False
        self.sim_thread.join(timeout=1.0)
        self.apply_control(0.0, 0.0)

        # --- Close log file safely ---
        if hasattr(self, "log_file"):
            self.log_file.close()
            self.get_logger().info(f"Saved MPC log to: {self.log_path}")

        if len(self.loop_durations) > 0:
            durations = np.array(self.loop_durations)
            avg_duration = np.mean(durations)
            min_d = np.min(durations)
            max_d = np.max(durations)
            self.get_logger().info(
                f"Average loop duration: {avg_duration:.4f}s "
                f"(min: {min_d:.4f}s, max: {max_d:.4f}s, avg freq: {1.0/avg_duration:.2f} Hz)"
            )

        self.get_logger().info("Controller shutdown complete.")



# ---------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------
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
