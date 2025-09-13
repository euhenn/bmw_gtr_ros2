#!/usr/bin/python3
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import scipy.linalg
from casadi import SX, vertcat, cos, sin, tan, arctan
from reference_trajectory_generation import TrajectoryGeneration

class MPC_KinematicBicycle:
    def __init__(self, dt_ocp=0.1, N_horizon=20, nodes=[73, 91]):
        self.lf = 0.13
        self.lr = 0.13
        self.L = self.lf + self.lr
        self.dt_ocp = dt_ocp
        self.N_horizon = N_horizon
        self.Tf = N_horizon * dt_ocp

        real_track = TrajectoryGeneration()
        self.trajectory, self.N = real_track.time_reference(N_horizon, nodes)
        
        # Ensure trajectory has 4 rows (x, y, psi, v)
        if self.trajectory.shape[0] < 4:
            # Add velocity row if missing
            self.trajectory = np.vstack([self.trajectory, np.zeros(self.trajectory.shape[1])])
        
        self.X0 = self.trajectory[:3, 0]  # Only take x, y, psi

        ocp = self._create_ocp()
        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    def _kinematic_model(self):
        x = SX.sym("x")
        y = SX.sym("y")
        psi = SX.sym("psi")
        x_var = vertcat(x, y, psi)

        v = SX.sym("v")
        delta = SX.sym("delta")
        u = vertcat(v, delta)

        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        psi_dot = SX.sym("psi_dot")
        xdot = vertcat(x_dot, y_dot, psi_dot)

        beta = arctan(self.lr * tan(delta) / self.L)
        f_expl = vertcat(v * cos(psi + beta),
                         v * sin(psi + beta),
                         v * sin(beta) / self.lr)
        f_impl = xdot - f_expl

        model = AcadosModel()
        model.name = "kinematic_bicycle"
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x_var
        model.xdot = xdot
        model.u = u

        return model

    def _create_ocp(self):
        ocp = AcadosOcp()
        model = self._kinematic_model()
        ocp.model = model

        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = 2  # Terminal cost only on x and y

        ocp.solver_options.N_horizon = self.N_horizon

        Q_mat = np.diag([1e2, 1e2, 1e2])
        R_mat = np.diag([1e0, 1e1])

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[:2],
                                        model.x[2] + arctan(self.lr * tan(model.u[1]) / self.L),
                                        model.u)

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag([1e2, 1e2])
        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.model.cost_y_expr_e = vertcat(model.x[:2])

        ocp.constraints.lbu = np.array([-2.0, -np.deg2rad(28)])
        ocp.constraints.ubu = np.array([2.0, np.deg2rad(28)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = self.Tf

        return ocp

    def set_initial_state(self, x0):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

    def set_reference(self, traj):
        # Ensure trajectory has 4 rows
        if traj.shape[0] < 4:
            # Add velocity row if missing
            traj = np.vstack([traj, np.zeros(traj.shape[1])])
        
        for j in range(self.N_horizon):
            if j < traj.shape[1]:
                yref = np.concatenate([traj[:3, j], [traj[3, j], 0.0]])
            else:
                # Use last point if trajectory is shorter than horizon
                yref = np.concatenate([traj[:3, -1], [traj[3, -1], 0.0]])
            self.solver.set(j, "yref", yref)

        # Terminal reference
        if self.N_horizon < traj.shape[1]:
            yref_e = traj[:2, self.N_horizon]
        else:
            yref_e = traj[:2, -1]
        self.solver.set(self.N_horizon, "yref", yref_e)

    def solve(self, state, traj):
        # Debug
        #print("DEBUG solve: state:", state)
        #print("DEBUG solve: traj shape:", traj.shape)

        self.set_initial_state(state)
        self.set_reference(traj)

        # show first few yrefs
        try:
            print("DEBUG yref[0]:", self.solver.get(0, "yref"))
            print("DEBUG yref[1]:", self.solver.get(1, "yref"))
        except Exception:
            pass

        # simple warm start: repeat previous u (or zeros) across horizon
        for j in range(self.N_horizon):
            try:
                self.solver.set(j, "u", np.array([self.v_cmd, 0.0]))
            except Exception:
                pass
            

        status = self.solver.solve()
        #print("DEBUG acados status:", status)

        u0 = self.solver.get(0, "u")
        #print("DEBUG u0:", u0)

        if status not in [0, 2]:
            raise RuntimeError(f"acados OCP solve failed with status {status}")

        return u0[0], u0[1]


    def get_reference_segment(self, idx):
        start = idx
        end = idx + self.N_horizon + 1
        if end > self.trajectory.shape[1]:
            end = self.trajectory.shape[1]
        print(f"DEBUG get_reference_segment: start={start}, end={end}, trajectory[:,start] = {self.trajectory[:,start]}")
        return self.trajectory[:, start:end]

if __name__ == "__main__":
    mpc = MPC_KinematicBicycle()
    x0 = np.array([0.0, 0.0, 0.0])
    
    # Debug: Check trajectory shape
    print(f"Trajectory shape: {mpc.trajectory.shape}")
    
    traj = mpc.get_reference_segment(0)
    print(f"Reference segment shape: {traj.shape}")
    
    v_cmd, delta_cmd = mpc.solve(x0, traj)
    print("First command:", v_cmd, delta_cmd)