#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from automobile_data_simulator import AutomobileDataSimulator
from mpc_kin import MPC_KinematicBicycle
import numpy as np
import curses
import time
from acados_template import AcadosOcpSolver

mpc_dt = 0.1
mpc_horizon = 20

class CarControllerNode(Node):
    def __init__(self, stdscr):
        super().__init__('car_controller_node')
        self.car = AutomobileDataSimulator(
            trig_control=True,
            trig_bno=True,
            trig_enc=True,
            trig_gps=True
        )

        # MPC setup
        self.mpc = MPC_KinematicBicycle(dt_ocp=mpc_dt, N_horizon=mpc_horizon)
        ocp = self.mpc.CreateOcpSolver_TimeKin()
        self.mpc_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_nonlinear.json")

        self.full_traj = self.mpc.trajectory
        self.traj_length = self.full_traj.shape[1]
        self.N_horizon = self.mpc.N_horizon

        self.prev_u = np.array([0.0, 0.0])
        self.control_interval = 0.1

        self.stdscr = stdscr
        curses.curs_set(0)

    def control_loop(self):
        idx = 0
        while rclpy.ok() and idx + self.N_horizon + 1 < self.traj_length:
            start_time = time.time()

            # Current state from simulator
            x, y = float(self.car.x_true), float(self.car.y_true)
            yaw_rad = np.deg2rad(float(self.car.yaw_true))
            x0 = np.array([x, y, yaw_rad])

            # Set reference trajectory for horizon
            traj_horizon = self.full_traj[:, idx:idx+self.N_horizon+1]
            for j in range(self.N_horizon):
                if idx + j < self.traj_length:
                    stage_ref = self.full_traj[:, idx + j]
                else:
                    stage_ref = self.full_traj[:, -1]
                yref = np.concatenate([stage_ref, np.array([0.0, 0.0])])
                self.mpc_solver.set(j, "yref", yref)

            # terminal yref
            idx_term = min(idx + self.N_horizon, self.traj_length - 1)
            yref_e = self.full_traj[:2, idx_term]
            self.mpc_solver.set(self.N_horizon, "yref", yref_e)

            # initial state constraint
            self.mpc_solver.set(0, "lbx", x0)
            self.mpc_solver.set(0, "ubx", x0)

            # Solve OCP
            try:
                status = self.mpc_solver.solve()
                if status not in [0, 2]:
                    commanded_v, commanded_delta = 0.0, 0.0
                else:
                    u0 = self.mpc_solver.get(0, "u")
                    commanded_v, commanded_delta = float(u0[0]), float(u0[1])
            except Exception:
                commanded_v, commanded_delta = 0.0, 0.0

            # Apply command
            self.car.drive_speed(commanded_v)
            self.car.drive_angle(np.rad2deg(commanded_delta))

            # Refresh terminal dashboard
            self.stdscr.erase()
            self.stdscr.addstr(0, 0, "Car Controller Dashboard")
            self.stdscr.addstr(2, 0, f"Position: ({x:.2f}, {y:.2f})")
            self.stdscr.addstr(3, 0, f"Yaw: {np.rad2deg(yaw_rad):.2f}°")
            self.stdscr.addstr(5, 0, f"Commanded Speed: {commanded_v:.2f} m/s")
            self.stdscr.addstr(6, 0, f"Commanded Steering: {np.rad2deg(commanded_delta):.2f}°")
            self.stdscr.addstr(8, 0, f"Trajectory Index: {idx}")
            self.stdscr.refresh()

            # Wait until next cycle
            elapsed = time.time() - start_time
            time.sleep(max(0.0, self.control_interval - elapsed))
            idx += 1

def main(stdscr):
    rclpy.init()
    node = CarControllerNode(stdscr)
    try:
        node.control_loop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    curses.wrapper(main)