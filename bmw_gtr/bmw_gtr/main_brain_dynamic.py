#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from automobile_data_simulator import AutomobileDataSimulator
#from mpc_kin_model import MPC_KinematicBicycle
from mpc_dyn_model import MPC_DynamicBicycle
from ref_time2spatial import *
import numpy as np
import time
import threading
import matplotlib.pyplot as plt

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
        self.ds = 0.1
        #self.mpc = MPC_KinematicBicycle(self.ds, N_horizon=20)
        self.mpc = MPC_DynamicBicycle(self.ds, N_horizon=10)
        self.control_thread = None
        self.pose_ready = False
        self.pose_ready_time = None

        # histories for offline plotting/inspection
        self.hist = {
            't': [],
            'x': [], 'y': [], 'yaw': [],
            'v': [], 'v_cmd': [],
            'a_cmd': [], 'delta_cmd': [],
            'epsi': [], 'ey': [], 'idx': [], 'solve_time': [],
            'x_ref': [], 'y_ref': []  # Added to store reference points
        }

    

        # Trajectory index
        self.idx = 0
        self.v_cmd = 0.25
        self.apply_control(self.v_cmd, 0.0)

        # Background simulator spin thread
        self._sim_running = True
        self.sim_thread = threading.Thread(target=self.spin_simulator, daemon=True)
        self.sim_thread.start()


    def spin_simulator(self):
        rate = 0.02
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            time.sleep(rate)

    def get_current_state(self):
        x = float(self.car.x_true)
        y = float(self.car.y_true)
        yaw = self.car.yaw_true #np.deg2rad(float(self.car.yaw_true))
        v = float(self.car.filtered_encoder_velocity)
        
        return np.array([x, y, yaw, v]) # self.v_cmd

    def apply_control(self, v, delta):
        self.car.drive_speed(v)
        self.car.drive_angle(np.rad2deg(delta))

    def find_closest_index(self, x, y):
        trajectory = self.mpc.trajectory
        diffs = trajectory[:2, :].T - np.array([x, y])
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    
    def run(self):
        i = 0 
        a_cmd = 0.0
        delta_cmd = 0.0
        #self.mpc_lock = threading.Lock()

        while (
            abs(self.car.x_true - 0.2) < 1e-3 and
            abs(self.car.y_true - 14.8) < 1e-3 and
            abs(self.car.yaw_true) < 1e-3
        ):
            self.get_logger().warn("Waiting for valid car pose...")
            time.sleep(0.2)


        prev_s = - self.ds  # track progress along path
        total_s = 0.0
        prev_x, prev_y = self.car.x_true, self.car.y_true
        prev_yaw = self.car.yaw_true

        yaw_hist = []
        yaw_ref_hist = []


        while rclpy.ok():
            state = self.get_current_state()
            x, y, yaw, v = state
            start_time = time.time()

            s_sim,epsi ,ey, self.idx =  time2spatial(x, y, yaw ,self.mpc.s_ref,self.mpc.trajectory[[2,3,6],:]) # y_ref will be replaced with self.mpc.trajectory
            #state_ocp = np.hstack((epsi,ey,state))

            vx = v
            vy = 0
            omega = (yaw - prev_yaw)/self.ds
            prev_yaw = yaw
            state_ocp = np.hstack((epsi,ey,x,y,vx,vy,yaw,omega))

            travelled_ds =  np.hypot(x - prev_x, y - prev_y)
            total_s += travelled_ds

            

            #self.get_logger().info(f"S {self.mpc.s_ref[i]} | Total S: ({total_s}) | previous S: {prev_s} ")
            
            if (s_sim - prev_s >= self.ds): #or (s_sim>self.mpc.s_ref[i-1])
                yaw_hist.append(yaw)
                self.get_logger().info(f"S ref {self.mpc.s_ref[self.idx]:.2f} | previous S: {prev_s:.2f} | S sim: {s_sim:.2f} ")
                #self.get_logger().info(f"car position ({self.car.x_true:.2f},{self.car.y_true:.2f}) yaw {self.car.yaw_true:.2f} vs traj start ({self.mpc.trajectory[2,i]:.2f},{self.mpc.trajectory[3, i]:.2f}) yaw {self.mpc.trajectory[4,i]:.2f}")
                self.get_logger().info(
                f"Errors: ({epsi:.3f}, {ey:.3f}) \n "
                f" Pos: ({x:.3f}, {y:.3f})\n "
                f"Yaw: {yaw:.2f} rad | Vel: {v:.2f} m/s \n")

            
                prev_s = s_sim
                prev_x, prev_y = x, y

                #i = i + 1
                #self.idx = i
                self.get_logger().info(f"Step {self.idx} | Trajectory: ({self.mpc.trajectory[:,self.idx+1]}) ")
            
                if i+1 >= self.mpc.trajectory.shape[1] - self.mpc.N_horizon - 1:  
                    self.get_logger().info("Stop condition reached")
                    self.apply_control(0.0, 0.0)
                    break

                try:
                    t0 = time.time()
                    # if v is zero, set small positive v_min for solver to avoid singularities
                    if abs(state_ocp[4]) < 1e-1:
                        state_ocp[4] = max(self.v_cmd, 0.4)  # choose small nonzero
                        time.sleep(0.2)
                    a_cmd, delta_cmd = self.mpc.solve(state_ocp, self.idx+1)
                    dt = time.time() - t0
                    solve_time = time.time() - t0
                    yaw_ref_hist.append(self.mpc.trajectory[6, self.idx])
                    self.get_logger().info(f"\ns_sim = {s_sim}\ns_ref = {self.mpc.s_ref[self.idx+1]}\n\n")
                except Exception as e:
                    self.get_logger().error(f"MPC solve failed: {e}")
                    # safe fallback
                    self.apply_control(0.0, 0.0)
                    break
                

                # 4) apply control
                #self.v_cmd = v + a_cmd *(time.time()-t0)
                self.v_cmd = self.mpc.trajectory[4, self.idx]
                self.apply_control(self.v_cmd, delta_cmd)
                #epsi, ey, x_s, y_s, yaw_s, v_s = state_ocp
                epsi, ey, x_s, y_s, vx_s, vy, yaw_s, omega = state_ocp
                self.get_logger().info(
                    f"v_cmd={self.v_cmd:.2f} | a_cmd={a_cmd:.2f} δ={np.rad2deg(delta_cmd):.2f}° | solve {dt:.3f}s")
   

                #if self.control_thread is None or not self.control_thread.is_alive():
                #    self.control_thread = threading.Thread(target=self.solve_and_apply, args=(state_ocp,))
                #    self.control_thread.start()
                
                
            # 7) save history for plotting / debug
            t_now = time.time() - start_time
            self.hist['t'].append(t_now)
            self.hist['x'].append(x); self.hist['y'].append(y); self.hist['yaw'].append(yaw)
            self.hist['v'].append(v); self.hist['v_cmd'].append(self.v_cmd)
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
                yaw_ref = self.mpc.trajectory[6, self.idx + 1]
            else:
                yaw_ref = np.nan

            
            self.get_logger().info(
                #f"\n--- Step idx={self.idx} ---"
                #f"\nPos: ({x:.2f}, {y:.2f}) | yaw={yaw:.3f} rad | v={v:.3f} m/s"
                f"\nErrors: epsi={epsi:.4f}, ey={ey:.4f} | s_sim={s_sim:.3f}"
                f"\nMPC -> a={a_cmd:.3f} m/s², δ={np.rad2deg(delta_cmd):.2f}° | self.v_cmd={self.v_cmd:.3f}"
                #f"\nSolve: {solve_time:.4f}s | traj_yaw_next={yaw_ref:.3f}"
            )
        
        
        yaw_hist = np.array(yaw_hist)
        yaw_ref_hist = np.array(yaw_ref_hist)
        plt.figure()
        plt.plot(yaw_hist, label='actual')
        plt.plot(yaw_ref_hist, label='ref')
        plt.legend()
        plt.show()

    def plot_history(self):
        # Enhanced plots for debugging including path tracking and MPC horizon
        try:
            t_raw = np.array(self.hist['t'])
            if t_raw.size == 0:
                self.get_logger().warn("No history to plot.")
                return

            # Use cumulative time so x-axis is monotonic in seconds
            t = np.cumsum(t_raw)
            
            # Convert histories to numpy arrays and handle length mismatches
            v = np.array(self.hist['v'])
            v_cmd = np.array(self.hist['v_cmd'])
            a_cmd = np.array(self.hist['a_cmd'])
            delta_cmd = np.array(self.hist['delta_cmd'])
            epsi = np.array(self.hist['epsi'])
            ey = np.array(self.hist['ey'])
            idx_hist = np.array(self.hist['idx'])

            # Ensure all arrays are the same length as time (trim or pad with NaN)
            def align_to_time(arr, tlen):
                arr = np.array(arr)
                if arr.size >= tlen:
                    return arr[:tlen]
                else:
                    # pad with NaNs
                    pad = np.full((tlen - arr.size,), np.nan)
                    return np.concatenate([arr, pad])

            n_t = t.size
            v = align_to_time(v, n_t)
            v_cmd = align_to_time(v_cmd, n_t)
            a_cmd = align_to_time(a_cmd, n_t)
            delta_cmd = align_to_time(delta_cmd, n_t)
            epsi = align_to_time(epsi, n_t)
            ey = align_to_time(ey, n_t)
            idx_hist = align_to_time(idx_hist, n_t)

            # Create 4 subplots
            plt.figure(figsize=(12, 10))

            # Plot 1: Velocity
            ax1 = plt.subplot(221)
            # only plot where neither x nor y is nan
            valid_v = ~np.isnan(v)
            valid_v_cmd = ~np.isnan(v_cmd)
            if np.any(valid_v):
                ax1.plot(t[valid_v], v[valid_v], '-', label='measured v', linewidth=2)
            if np.any(valid_v_cmd):
                ax1.plot(t[valid_v_cmd], v_cmd[valid_v_cmd], '--', label='cmd v', linewidth=2)
            ax1.set_ylabel('v (m/s)')
            ax1.set_xlabel('Time (s)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Velocity Tracking')

            # Plot 2: Acceleration and Steering
            ax2 = plt.subplot(222)
            valid_a = ~np.isnan(a_cmd)
            if np.any(valid_a):
                ln1 = ax2.plot(t[valid_a], a_cmd[valid_a], '-', label='a_cmd', linewidth=2)
            else:
                ln1 = []
            ax2.set_ylabel('a (m/s^2)')
            ax2.set_xlabel('Time (s)')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Acceleration & Steering Commands')

            # Plot 2b: Steering (twin axis)
            ax2b = ax2.twinx()
            valid_delta = ~np.isnan(delta_cmd)
            steer_deg = np.rad2deg(delta_cmd)
            if np.any(valid_delta):
                ln2 = ax2b.plot(t[valid_delta], steer_deg[valid_delta], '-', label='steer (deg)', linewidth=1, alpha=0.85)
            else:
                ln2 = []
            ax2b.set_ylabel('Steering (deg)')

            # Combine legends from both axes
            '''
            lines = []
            labels = []
            for ln in (ln1 if isinstance(ln1, list) else [ln1]):
                if ln:
                    for l in ln:
                        lines.append(l)
                        labels.append(l.get_label())
            for ln in (ln2 if isinstance(ln2, list) else [ln2]):
                if ln:
                    for l in ln:
                        lines.append(l)
                        labels.append(l.get_label())
            if lines:
                ax2.legend(lines, labels, loc='upper right')
            '''
            # Plot 3: Errors
            ax3 = plt.subplot(223)
            valid_epsi = ~np.isnan(epsi)
            valid_ey = ~np.isnan(ey)
            if np.any(valid_epsi):
                ax3.plot(t[valid_epsi], epsi[valid_epsi], '-', label='epsi (heading error)', linewidth=2)
            if np.any(valid_ey):
                ax3.plot(t[valid_ey], ey[valid_ey], '-', label='ey (lateral error)', linewidth=2)
            ax3.set_ylabel('Errors')
            ax3.set_xlabel('Time (s)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Tracking Errors')

            # Plot 4: Path tracking with MPC horizon
            ax4 = plt.subplot(224)
            
            # Full reference trajectory
            traj_x = self.mpc.trajectory[2, :]
            traj_y = self.mpc.trajectory[3, :]
            ax4.plot(traj_x, traj_y, 'k-', label='Full Reference', linewidth=2, alpha=0.5)
            
            # Actual path traveled
            if len(self.hist['x']) > 0:
                ax4.plot(self.hist['x'], self.hist['y'], 'r-', label='Actual Path', linewidth=2)
            
            # Plot MPC horizon for the last few steps to see evolution
            num_horizon_plots = min(5, len(self.hist['idx']))  # Show last 5 horizons
            for i in range(-num_horizon_plots, 0):
                try:
                    idx = int(self.hist['idx'][i])
                except Exception:
                    continue
                if idx + self.mpc.N_horizon < len(traj_x) and idx >= 0:
                    # Extract the horizon that MPC was optimizing over at this step
                    horizon_x = traj_x[idx:idx+self.mpc.N_horizon]
                    horizon_y = traj_y[idx:idx+self.mpc.N_horizon]
                    
                    # Color based on how recent it is
                    alpha = 0.3 + 0.7 * (i + num_horizon_plots) / num_horizon_plots
                    color = plt.cm.viridis((i + num_horizon_plots) / num_horizon_plots)
                    
                    ax4.plot(horizon_x, horizon_y, '--', color=color, 
                            label=f'MPC horizon (step {len(self.hist["idx"])+i})' if i == -1 else "",
                            alpha=alpha, linewidth=1.5)
                    
                    # Mark the current reference point for this horizon
                    ax4.plot(traj_x[idx], traj_y[idx], 'o', color=color, markersize=4, alpha=alpha)
            
            # Highlight the final MPC horizon (most recent)
            if len(self.hist['idx']) > 0:
                final_idx = int(self.hist['idx'][-1])
                if 0 <= final_idx + self.mpc.N_horizon < len(traj_x):
                    final_horizon_x = traj_x[final_idx:final_idx+self.mpc.N_horizon]
                    final_horizon_y = traj_y[final_idx:final_idx+self.mpc.N_horizon]
                    ax4.plot(final_horizon_x, final_horizon_y, 'g--', linewidth=2, 
                            label='Final MPC Horizon', alpha=0.8)
                    ax4.plot(traj_x[final_idx], traj_y[final_idx], 'go', markersize=6, 
                            label='Final Ref Start')

            # Start and end markers
            if len(self.hist['x']) > 0:
                ax4.plot(self.hist['x'][0], self.hist['y'][0], 'go', markersize=8, label='Start')
                ax4.plot(self.hist['x'][-1], self.hist['y'][-1], 'ro', markersize=8, label='End')
                
                # Plot car orientation at final position
                final_yaw = self.hist['yaw'][-1]
                dx = 0.3 * np.cos(final_yaw)
                dy = 0.3 * np.sin(final_yaw)
                ax4.arrow(self.hist['x'][-1], self.hist['y'][-1], dx, dy, 
                         head_width=0.1, head_length=0.1, fc='r', ec='r')

            ax4.set_xlabel('x (m)')
            ax4.set_ylabel('y (m)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axis('equal')
            ax4.set_title('Path Tracking with MPC Horizons')
            
            # Add text box with performance metrics
            if len(self.hist['ey']) > 0:
                max_lateral_error = np.nanmax(np.abs(ey))
                avg_lateral_error = np.nanmean(np.abs(ey))
                max_heading_error = np.nanmax(np.abs(epsi))
                textstr = f'Max lateral error: {max_lateral_error:.3f} m\nAvg lateral error: {avg_lateral_error:.3f} m\nMax heading error: {np.rad2deg(max_heading_error):.2f}°'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, 
                              fontsize=9, verticalalignment='top', bbox=props)

            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f"Plotting failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def shutdown(self):
        self.plot_history() 
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
