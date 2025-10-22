#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import numpy as np
from time import time, sleep
import threading
import matplotlib.pyplot as plt
import cv2 as cv

from automobile_data_simulator import AutomobileDataSimulator
from mpc_kin_model_eugen import MPC_KinematicBicycle
from ref_time2spatial import time2spatial

# Control loop settings
TARGET_FPS = 10.0           # run control loop at this frequency
DT = 1.0 / TARGET_FPS       # fixed control timestep used to integrate acceleration

# Vehicle constraints
DS = 0.05       # spatial step [m]
N_HORIZON = 70  # prediction horizon
#NODES = [73, 97, 100, 130, 140]  # waypoints for path planning
NODES = [73, 97, 100, 137]

class CarControllerNode(Node):
    def __init__(self):
        super().__init__('car_controller_node')

        # === Simulator setup ===
        self.car = AutomobileDataSimulator(
            trig_control=True,
            trig_bno=True,
            trig_enc=True,
            trig_gps=True
        )

        # === MPC setup ===
        self.mpc = MPC_KinematicBicycle(DS, N_HORIZON, NODES)

        # initial commanded velocity (simulator driven by velocity)
        self.v_cmd = 0.1
        self.apply_control(self.v_cmd, 0.0)

        self.idx = 0
        self._sim_running = True

        # histories for offline plotting/inspection
        self.hist = {
            't': [],
            'x': [], 'y': [], 'yaw': [],
            'v': [], 'v_cmd': [],
            'a_cmd': [], 'delta_cmd': [],
            'epsi': [], 'ey': [], 'idx': [], 'solve_time': [],
            'x_ref': [], 'y_ref': []  # Added to store reference points
        }

        # background simulator thread
        self.sim_thread = threading.Thread(target=self.spin_simulator, daemon=True)
        self.sim_thread.start()

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------
    def spin_simulator(self):
        """ Continuously update the car simulator. """
        rate = 0.02  # 50 Hz simulator update
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            sleep(rate)

    def get_current_state(self):
        """ Returns current [x, y, yaw, v] from simulator. """
        x = float(self.car.x_true)
        y = float(self.car.y_true)
        yaw = float(self.car.yaw_true)
        v = float(self.car.filtered_encoder_velocity)
        return np.array([x, y, yaw, v])

    def apply_control(self, v, delta):
        """ Apply control commands (velocity [m/s] and steer [rad]) to the simulator. """
        # clamp before sending ; NO NEED TO CLAMP
        #v = float(np.clip(v, MIN_SPEED, MAX_SPEED))
        #delta = float(np.clip(delta, -MAX_STEER_RAD, MAX_STEER_RAD))
        # simulator expects speed and steering angle in degrees
        self.car.drive_speed(v)
        self.car.drive_angle(np.rad2deg(delta))



    # ----------------------------------------------------------
    # Main control loop
    # ----------------------------------------------------------
    def run(self):

        self.get_logger().info("Starting MPC control loop...")
        start_time = time()

        while rclpy.ok():
            loop_start = time()

            # 1) read state
            state = self.get_current_state()
            x, y, yaw, v = state

            # 2) spatial mapping -> s_sim, epsi, ey, idx
            try:
                s_sim, epsi, ey, self.idx = time2spatial(
                    x, y, yaw, self.mpc.s_ref, self.mpc.trajectory[2:5, :]
                )
            except Exception as e:
                self.get_logger().error(f"Spatial mapping failed: {e}")
                break

            state_ocp = np.hstack((epsi, ey, state))

            # 3) MPC solve
            try:
                t0 = time()
                # guard against v ~ 0 for solver
                v_for_solver = v if abs(v) > 1e-3 else max(self.v_cmd, 0.2)
                state_ocp[-1] = v_for_solver

                a_cmd, delta_cmd = self.mpc.solve(state_ocp, self.idx + 1)
                solve_time = time() - t0

            except Exception as e:
                self.get_logger().error(f"MPC solve failed: {e}")
                self.apply_control(0.0, 0.0)
                break

            # 4) sanitize commands
            #a_cmd, delta_cmd = self.safe_clamp_cmds(a_cmd, delta_cmd)

            # 5) integrate acceleration using fixed dt (not solver time)
            v_next = v + a_cmd * DT
            # clamp velocity
            #v_next = float(np.clip(v_next, MIN_SPEED, MAX_SPEED))

            # 6) apply
            self.apply_control(v_next, delta_cmd)

            # 7) save history for plotting / debug
            t_now = time() - start_time
            self.hist['t'].append(t_now)
            self.hist['x'].append(x); self.hist['y'].append(y); self.hist['yaw'].append(yaw)
            self.hist['v'].append(v); self.hist['v_cmd'].append(v_next)
            self.hist['a_cmd'].append(a_cmd); self.hist['delta_cmd'].append(delta_cmd)
            self.hist['epsi'].append(epsi); self.hist['ey'].append(ey)
            self.hist['idx'].append(self.idx); self.hist['solve_time'].append(solve_time)
            
            # Store current reference point for plotting
            if 0 <= self.idx < self.mpc.trajectory.shape[1]:
                self.hist['x_ref'].append(self.mpc.trajectory[2, self.idx])
                self.hist['y_ref'].append(self.mpc.trajectory[3, self.idx])
            else:
                self.hist['x_ref'].append(np.nan)
                self.hist['y_ref'].append(np.nan)

            # 8) logging - compact and informative
            # protect indexing into trajectory when idx+1 near end
            yaw_ref = None
            if 0 <= (self.idx + 1) < self.mpc.trajectory.shape[1]:
                yaw_ref = self.mpc.trajectory[4, self.idx + 1]
            else:
                yaw_ref = np.nan

            
            self.get_logger().info(
                #f"\n--- Step idx={self.idx} ---"
                #f"\nPos: ({x:.2f}, {y:.2f}) | yaw={yaw:.3f} rad | v={v:.3f} m/s"
                f"\nErrors: epsi={epsi:.4f}, ey={ey:.4f} | s_sim={s_sim:.3f}"
                f"\nMPC -> a={a_cmd:.3f} m/s², δ={np.rad2deg(delta_cmd):.2f}° | v_next={v_next:.3f}"
                #f"\nSolve: {solve_time:.4f}s | traj_yaw_next={yaw_ref:.3f}"
            )
            """
            self.get_logger().info(
                f"\n--- filtered_encoder_velocity = {self.car.filtered_encoder_velocity:.3f} , v={v:.3f} , v_for_solver={v_for_solver:.3f} ---"
                f"\n--- v_next = {v:.3f} + {a_cmd:.3f} * {DT:.3f} = {v_next:.3f} ---"
            )
            """
            # Watch for sudden yaw reference jumps (common cause of big steer)
            if len(self.hist['yaw']) >= 2:
                if abs(yaw_ref - self.hist['yaw'][-1]) > np.deg2rad(20):  # threshold
                    self.get_logger().warn(
                        f"Large yaw ref jump detected at idx={self.idx}: ref_yaw={yaw_ref:.3f}"
                    )

            # 9) stop condition - safe indexing check
            if self.idx >= (self.mpc.trajectory.shape[1] - N_HORIZON - 1):
                self.get_logger().info("Reached end of trajectory: stopping.")
                self.apply_control(0.0, 0.0)
                break

            # 10) timing - enforce target FPS (sleep remainder)
            loop_end = time()
            loop_duration = loop_end - loop_start
            target_period = 1.0 / TARGET_FPS
            if loop_duration < target_period:
                sleep(target_period - loop_duration)
            else:
                self.get_logger().warn(
                    f"Control loop overrun: took {loop_duration:.3f}s > {target_period:.3f}s"
                )

        # plot results after loop ends
        #self.plot_history() # moved the plotting to when i stop the simulation

    # ----------------------------------------------------------
    def plot_history(self):
        # Enhanced plots for debugging including path tracking and MPC horizon
        try:
            t = np.array(self.hist['t'])
            
            # Create 4 subplots
            plt.figure(figsize=(12, 10))
            
            # Plot 1: Velocity
            plt.subplot(221)
            plt.plot(t, self.hist['v'], 'b-', label='measured v', linewidth=2)
            plt.plot(t, self.hist['v_cmd'], 'r--', label='cmd v', linewidth=2)
            plt.ylabel('v (m/s)')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Velocity Tracking')

            # Plot 2: Acceleration and Steering
            plt.subplot(222)
            plt.plot(t, self.hist['a_cmd'], 'g-', label='a_cmd', linewidth=2)
            plt.ylabel('a (m/s^2)')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Acceleration Commands')
            
            # Plot 2b: Steering (twin axis)
            ax2b = plt.gca().twinx()
            ax2b.plot(t, np.rad2deg(self.hist['delta_cmd']), 'm-', label='steer (deg)', linewidth=1, alpha=0.7)
            ax2b.set_ylabel('Steering (deg)')
            ax2b.legend(loc='upper right')

            # Plot 3: Errors
            plt.subplot(223)
            plt.plot(t, self.hist['epsi'], 'r-', label='epsi (heading error)', linewidth=2)
            plt.plot(t, self.hist['ey'], 'b-', label='ey (lateral error)', linewidth=2)
            plt.ylabel('Errors')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Tracking Errors')

            # Plot 4: Path tracking with MPC horizon
            plt.subplot(224)
            
            # Full reference trajectory
            traj_x = self.mpc.trajectory[2, :]
            traj_y = self.mpc.trajectory[3, :]
            plt.plot(traj_x, traj_y, 'k-', label='Full Reference', linewidth=2, alpha=0.5)
            
            # Actual path traveled
            plt.plot(self.hist['x'], self.hist['y'], 'r-', label='Actual Path', linewidth=2)
            
            # Plot MPC horizon for the last few steps to see evolution
            num_horizon_plots = min(5, len(self.hist['idx']))  # Show last 5 horizons
            for i in range(-num_horizon_plots, 0):
                idx = self.hist['idx'][i]
                if idx + N_HORIZON < len(traj_x):
                    # Extract the horizon that MPC was optimizing over at this step
                    horizon_x = traj_x[idx:idx+N_HORIZON]
                    horizon_y = traj_y[idx:idx+N_HORIZON]
                    
                    # Color based on how recent it is
                    alpha = 0.3 + 0.7 * (i + num_horizon_plots) / num_horizon_plots
                    color = plt.cm.viridis((i + num_horizon_plots) / num_horizon_plots)
                    
                    plt.plot(horizon_x, horizon_y, '--', color=color, 
                            label=f'MPC horizon (step {len(self.hist["idx"])+i})' if i == -1 else "",
                            alpha=alpha, linewidth=1.5)
                    
                    # Mark the current reference point for this horizon
                    plt.plot(traj_x[idx], traj_y[idx], 'o', color=color, markersize=4, alpha=alpha)
            
            # Highlight the final MPC horizon (most recent)
            if len(self.hist['idx']) > 0:
                final_idx = self.hist['idx'][-1]
                if final_idx + N_HORIZON < len(traj_x):
                    final_horizon_x = traj_x[final_idx:final_idx+N_HORIZON]
                    final_horizon_y = traj_y[final_idx:final_idx+N_HORIZON]
                    plt.plot(final_horizon_x, final_horizon_y, 'g--', linewidth=2, 
                            label='Final MPC Horizon', alpha=0.8)
                    plt.plot(traj_x[final_idx], traj_y[final_idx], 'go', markersize=6, 
                            label='Final Ref Start')

            # Start and end markers
            if len(self.hist['x']) > 0:
                plt.plot(self.hist['x'][0], self.hist['y'][0], 'go', markersize=8, label='Start')
                plt.plot(self.hist['x'][-1], self.hist['y'][-1], 'ro', markersize=8, label='End')
                
                # Plot car orientation at final position
                final_yaw = self.hist['yaw'][-1]
                dx = 0.3 * np.cos(final_yaw)
                dy = 0.3 * np.sin(final_yaw)
                plt.arrow(self.hist['x'][-1], self.hist['y'][-1], dx, dy, 
                         head_width=0.1, head_length=0.1, fc='r', ec='r')

            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.title('Path Tracking with MPC Horizons')
            
            # Add text box with performance metrics
            if len(self.hist['ey']) > 0:
                max_lateral_error = np.max(np.abs(self.hist['ey']))
                avg_lateral_error = np.mean(np.abs(self.hist['ey']))
                max_heading_error = np.max(np.abs(self.hist['epsi']))
                textstr = f'Max lateral error: {max_lateral_error:.3f} m\nAvg lateral error: {avg_lateral_error:.3f} m\nMax heading error: {np.rad2deg(max_heading_error):.2f}°'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                              fontsize=9, verticalalignment='top', bbox=props)

            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f"Plotting failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    # ----------------------------------------------------------
    def shutdown(self):
        self.plot_history() # plot at the end, before shutting down
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