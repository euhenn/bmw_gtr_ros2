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
        self.ds = 0.01
        self.mpc = MPC_KinematicBicycle(self.ds, N_horizon=10)
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
        self.rate = 0.02
        while self._sim_running and rclpy.ok():
            rclpy.spin_once(self.car, timeout_sec=0.0)
            time.sleep(self.rate)

    def get_current_state(self):
        x = float(self.car.x_true)
        y = float(self.car.y_true)
        yaw = self.car.yaw_true 
        v = float(self.car.filtered_encoder_velocity)
        
        return np.array([x, y, yaw, v])

    def apply_control(self, v, delta):
        self.car.drive_speed(v)
        self.car.drive_angle(np.rad2deg(delta))

    def find_closest_index(self, x, y):
        trajectory = self.mpc.trajectory
        diffs = trajectory[:2, :].T - np.array([x, y])
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    
    def run(self):
        a_cmd = 0.0
        delta_cmd = 0.0

        while (abs(self.car.x_true - 0.2) < 1e-3 and abs(self.car.y_true - 14.8) < 1e-3 and abs(self.car.yaw_true) < 1e-3):
            self.get_logger().warn("Waiting for valid car pose...")
            time.sleep(0.2)


        prev_s = - self.ds  
        prev_x, prev_y = self.car.x_true, self.car.y_true

        yaw_hist = []
        yaw_ref_hist = []

        v_hist = []
        v_ref_hist = []

        xhist = []
        yhist = []


        while rclpy.ok():
            state = self.get_current_state()
            prev_s = s_sim
            prev_x, prev_y = x, y

            x, y, yaw, v = state
            s_sim,state_ocp, self.idx, error_dist = self.mpc.ClosestPoint(state) #time2spatial(x, y, yaw ,self.mpc.s_ref,self.mpc.trajectory[2:5,:]) # y_ref will be replaced with self.mpc.trajectory
            state_ocp = np.hstack((epsi,ey,state))
            
            yaw_hist.append(yaw)
            xhist.append(x)
            yhist.append(y)
            v_hist.append(v)

            self.get_logger().info(f"Step {self.idx}  S {self.mpc.s_ref[self.idx]} | previous S: {prev_s} ")
            
            self.get_logger().info(
            f"Errors: ({epsi:.3f}, {ey:.3f}) \n"
            f" Pos: ({x:.3f}, {y:.3f})| Trajectory Position ({self.mpc.trajectory[2, self.idx]:.3f}, {self.mpc.trajectory[3, self.idx]:.3f}) \n " 
            f"Yaw: {yaw:.2f} rad | Desired Yaw: {self.mpc.trajectory[4, self.idx]:.2f} rad  \n"
            f"Vel: {v:.2f} m/s | ")

    
            
            if i+1 >= self.mpc.trajectory.shape[1] - self.mpc.N_horizon - 1:  
                self.get_logger().info("Stop condition reached")
                self.apply_control(0.0, 0.0)
                break

            try:
                t0 = time.time()
                if abs(state_ocp[-1]) < 1e-3:
                    state_ocp[-1] = max(self.v_cmd, 0.2)  # choose small nonzero
                    time.sleep(0.2)

                a_cmd, delta_cmd = self.mpc.solve(state_ocp, self.idx+1)
                
                dt = time.time() - t0

                yaw_ref_hist.append(self.mpc.trajectory[4, self.idx+1])
                v_ref_hist.append(self.mpc.trajectory[5, self.idx+1])
        
            except Exception as e:
                self.get_logger().error(f"MPC solve failed: {e}")
                # safe fallback
                self.apply_control(0.0, 0.0)
                break
                

                # 4) apply control
                self.v_cmd = v + a_cmd * self.rate
                #self.v_cmd = self.mpc.trajectory[-1,self.idx]
                self.apply_control(self.v_cmd, delta_cmd)
                epsi, ey, x_s, y_s, yaw_s, v_s = state_ocp
                self.get_logger().info(
                    f"v_cmd={self.v_cmd:.2f} | a_cmd={a_cmd:.2f} δ={np.rad2deg(delta_cmd):.2f}° | solve {dt:.3f}s")
 

        
        yaw_hist = np.array(yaw_hist)
        yaw_ref_hist = np.array(yaw_ref_hist)
        plt.figure()
        plt.plot(yaw_hist,'r-', label='actual')
        plt.plot(yaw_ref_hist, 'b-',label='ref')
        plt.legend()
        plt.title("Yaw Angle")
        plt.grid(True, alpha=0.3)
        plt.show()

        v_hist = np.array(v_hist)
        v_ref_hist = np.array(v_ref_hist)
        plt.figure()
        plt.plot(v_hist,'r-', label='actual')
        plt.plot(v_ref_hist,'b-', label='ref')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("Speed")
        plt.show()


        plt.figure()
        traj_x = self.mpc.trajectory[2, :]
        traj_y = self.mpc.trajectory[3, :]
        plt.plot(traj_x, traj_y, 'b-', label='Full Reference', linewidth=2, alpha=0.5)
        # Actual path traveled
        plt.plot(xhist, yhist, 'r-', label='Actual Path', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        

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
