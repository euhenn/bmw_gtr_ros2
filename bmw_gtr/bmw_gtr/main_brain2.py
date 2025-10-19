#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from automobile_data_simulator import AutomobileDataSimulator
from mpc_kin_model import MPC_KinematicBicycle
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
        self.ds = 0.06
        self.mpc = MPC_KinematicBicycle(self.ds, N_horizon=40)
        self.control_thread = None
        self.pose_ready = False
        self.pose_ready_time = None

    

        # Trajectory index
        self.idx = 0
        self.v_cmd = 0.5
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

        yaw_hist = []
        yaw_ref_hist = []


        while rclpy.ok():
            state = self.get_current_state()
            x, y, yaw, v = state#self.mpc.trajectory[2:,i]
            s_sim,epsi ,ey, self.idx =  time2spatial(x, y, yaw ,self.mpc.s_ref,self.mpc.trajectory[2:5,:]) # y_ref will be replaced with self.mpc.trajectory
            state_ocp = np.hstack((epsi,ey,state))

            travelled_ds =  np.hypot(x - prev_x, y - prev_y)
            total_s += travelled_ds

            

            #self.get_logger().info(f"S {self.mpc.s_ref[i]} | Total S: ({total_s}) | previous S: {prev_s} ")
            
            if (s_sim - prev_s >= self.ds): #or (s_sim>self.mpc.s_ref[i-1])
                yaw_hist.append(yaw)
                self.get_logger().info(f"S ref {self.mpc.s_ref[self.idx]:.2f} | previous S: {prev_s:.2f} | S sim: {s_sim:.2f} ")
                #self.get_logger().info(f"car position ({self.car.x_true:.2f},{self.car.y_true:.2f}) yaw {self.car.yaw_true:.2f} vs traj start ({self.mpc.trajectory[2,i]:.2f},{self.mpc.trajectory[3, i]:.2f}) yaw {self.mpc.trajectory[4,i]:.2f}")
                self.get_logger().info(
                f"Errors: ({epsi:.3f}, {ey:.3f}) "
                f"Step {self.idx} | Pos: ({x:.3f}, {y:.3f}) "
                f"Yaw: {yaw:.2f} rad | Vel: {v:.2f} m/s | ")

            
                prev_s = s_sim
                prev_x, prev_y = x, y

                #i = i + 1
                #self.idx = i
                self.get_logger().info(f"Step {self.idx} | Trajectory: ({self.mpc.trajectory[:,self.idx]}) ")
            
                if i+1 >= self.mpc.trajectory.shape[1] - self.mpc.N_horizon - 1:  
                    self.get_logger().info("Stop condition reached")
                    self.apply_control(0.0, 0.0)
                    break

                try:
                    t0 = time.time()
                    # if v is zero, set small positive v_min for solver to avoid singularities
                    if abs(state_ocp[-1]) < 1e-3:
                        state_ocp[-1] = max(self.v_cmd, 0.5)  # choose small nonzero
                        time.sleep(0.2)
                    a_cmd, delta_cmd = self.mpc.solve(state_ocp, self.idx+1)
                    dt = time.time() - t0
                    yaw_ref_hist.append(self.mpc.trajectory[4, self.idx+1])
                    self.get_logger().info(f"\ns_sim = {s_sim}\ns_ref = {self.mpc.s_ref[self.idx+1]}\n\n")
                except Exception as e:
                    self.get_logger().error(f"MPC solve failed: {e}")
                    # safe fallback
                    self.apply_control(0.0, 0.0)
                    break
                

                # 4) apply control
                self.v_cmd = v + a_cmd *(time.time()-t0)
                self.apply_control(self.v_cmd, delta_cmd)
                epsi, ey, x_s, y_s, yaw_s, v_s = state_ocp
                self.get_logger().info(
                    f"v_cmd={self.v_cmd:.2f} | a_cmd={a_cmd:.2f} δ={np.rad2deg(delta_cmd):.2f}° | solve {dt:.3f}s")
   

                #if self.control_thread is None or not self.control_thread.is_alive():
                #    self.control_thread = threading.Thread(target=self.solve_and_apply, args=(state_ocp,))
                #    self.control_thread.start()
                
                self.get_logger().info(f"Number of iteration passed{i}")
        
        
        yaw_hist = np.array(yaw_hist)
        yaw_ref_hist = np.array(yaw_ref_hist)
        plt.figure()
        plt.plot(yaw_hist, label='actual')
        plt.plot(yaw_ref_hist, label='ref')
        plt.legend()
        plt.show()


    '''  
    def solve_and_apply(self, state_ocp):
        t0 = time.time()
        epsi ,ey, x, y, yaw, v = state_ocp 
        print("state_ocp:", state_ocp)
        print("shape:", state_ocp.shape)
        a_cmd, delta_cmd = self.mpc.solve(state_ocp, self.idx)
        self.get_logger().info(f"MPC solve took {time.time()-t0:.3f}s")
        self.a_cmd, delta_cmd = float(a_cmd), float(delta_cmd)
        self.v_cmd = v + a_cmd *(time.time()-t0) 
        self.apply_control(self.v_cmd, delta_cmd)

        
        self.get_logger().info(
                f"Step {self.idx} | Pos: ({x:.2f}, {y:.2f}) "
                f"Yaw: {np.rad2deg(yaw):.2f}° | Vel: {self.v_cmd:.2f} m/s | "
                f"Cmd -> a: {self.a_cmd:.2f} m/s, δ: {np.rad2deg(delta_cmd):.1f}°")
    '''

    def solve_and_apply(self, state_ocp):
        try:
            with self.mpc_lock:
                t0 = time.time()
                a_cmd, delta_cmd = self.mpc.solve(state_ocp, self.idx)
                dt = time.time() - t0

            # === 4. Apply control ===
            self.apply_control(self.v_cmd, delta_cmd)

            epsi, ey, x, y, yaw, v = state_ocp
            self.get_logger().info(
                f"MPC OK | Step {self.idx} | Pos ({x:.2f}, {y:.2f}) "
                f"Yaw {np.rad2deg(yaw):.2f}° | Vel {v:.2f} m/s | "
                f"Cmd a={a_cmd:.2f}, δ={np.rad2deg(delta_cmd):.1f}° | "
                f"Solve {dt:.3f}s"
            )

        except RuntimeError as e:
            self.get_logger().error(f"MPC solve failed (RuntimeError): {e}")
            self.apply_control(1.0, 0.0)

        except Exception as e:
            self.get_logger().error(f"MPC thread exception: {e}")
            self.apply_control(0.0, 0.0)


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
