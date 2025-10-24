#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import matplotlib.pyplot as plt

from automobile_data_simulator import AutomobileDataSimulator
from mpc_controller import MPC_KinematicBicycle

# -------------------------------
# Control loop parameters
# -------------------------------
TARGET_FPS = 15.0         # target loop frequency [Hz]
DT = 1.0 / TARGET_FPS     # fixed timestep for velocity integration [s]


class CarControllerNode(Node):
    def __init__(self):
        super().__init__('car_controller_node')
        self.get_logger().info("Initializing Car Controller Node...")

        # --- Initialize simulator ---
        self.car = AutomobileDataSimulator(
            trig_control=True,
            trig_bno=True,
            trig_enc=True,
            trig_gps=True
        )

        # --- Background simulator thread ---
        self._sim_running = True
        self.sim_thread = threading.Thread(target=self.spin_simulator, daemon=True)
        self.sim_thread.start()
        self.car.drive_speed(0.0)

        # --- Initialize MPC ---
        self.mpc = MPC_KinematicBicycle(ds=0.05, N_horizon=40)

        # --- Initialize control variables ---
        self.v_cmd = 0.4

        self.get_logger().info(f"Control loop target rate: {TARGET_FPS} Hz")

    # ---------------------------------------------------------------
    # SIMULATION THREAD
    # ---------------------------------------------------------------
    def spin_simulator(self):
        """Continuously update simulator data in background."""
        rate = 0.01  # 100 Hz simulator update
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            time.sleep(rate)

    # ---------------------------------------------------------------
    # STATE ACCESSORS
    # ---------------------------------------------------------------
    def get_current_state(self):
        """Return [x, y, yaw, v]."""
        x = float(self.car.x_true)
        y = float(self.car.y_true)
        yaw = float(self.car.yaw_true)
        v = float(self.car.filtered_encoder_velocity)
        return np.array([x, y, yaw, v])

    def apply_control(self, v, delta):
        """Send control commands to simulator."""
        self.car.drive_speed(v)
        self.car.drive_angle(np.rad2deg(delta))

    # ---------------------------------------------------------------
    # MAIN CONTROL LOOP (WITH FPS LIMITER)
    # ---------------------------------------------------------------
    def run(self):
        self.get_logger().info("Starting main control loop...")
        a_prev, delta_prev = 0.0, 0.0
        while rclpy.ok():
            loop_start = time.time()  # start timing

            try:
                # === 1. Get current state ===
                x, y, yaw, v = self.get_current_state()

                # === 2. Transform to spatial coordinates ===
                state_ocp, idx = self.mpc.get_state(x, y, yaw, v)
                e_psi, e_y = state_ocp[0], state_ocp[1]


                # === 3. Solve MPC ===
                a_cmd, delta_cmd = self.mpc.solve(state_ocp, idx + 1, a_prev, delta_prev)

                # === 4. Integrate control & apply ===
                v_next = max(0.0, v + a_cmd * DT)
                self.apply_control(v_next, delta_cmd)
                a_prev, delta_prev = a_cmd, delta_cmd
                self.v_cmd = v_next

                # === 5. Log info ===
                #print(f"v={v:.3f} → {v_next:.3f} | a={a_cmd:.3f} m/s2 | δ={np.rad2deg(delta_cmd):.2f}°")
                #print(f"idx={idx} | e_y={e_y:.4f} m | e_psi={np.rad2deg(e_psi):.2f}°")

            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")
                self.apply_control(0.0, 0.0)
                break

            # === 6. FPS limiter ===
            loop_duration = time.time() - loop_start
            target_period = 1.0 / TARGET_FPS
            if loop_duration < target_period:
                time.sleep(target_period - loop_duration)
            else:
                self.get_logger().warn(
                    f"Loop overrun: took {loop_duration:.3f}s > {target_period:.3f}s"
                )

    # ---------------------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------------------
    def shutdown(self):
        """Graceful shutdown."""
        self._sim_running = False
        self.sim_thread.join(timeout=1.0)
        self.apply_control(0.0, 0.0)
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
