#!/usr/bin/python3
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import scipy.linalg
from casadi import SX, vertcat, cos, sin, tan, arctan, interpolant
from reference_generation_velocity import TrajectoryGeneration
from ref_time2spatial import time2spatial


class MPC_KinematicBicycle:
    def __init__(self, ds=0.05, N_horizon=50, nodes=[73, 91, 125]):
        """Initialize MPC with kinematic bicycle spatial model."""
        # vehicle parameters
        self.lf = 0.13
        self.lr = 0.13
        self.L = self.lf + self.lr

        # MPC parameters
        self.ds = ds
        self.N_horizon = N_horizon
        self.Tf = self.N_horizon * self.ds

        # Reference trajectory generation
        traj_gen = TrajectoryGeneration(self.ds, self.N_horizon)
        self.trajectory, self.s_ref, kappa_ref = traj_gen.generating_spatial_reference(nodes)
        self.kappa = interpolant("kappa", "bspline", [self.s_ref], kappa_ref)
        self.X0 = self.trajectory[:, 0]

        # Create solver
        ocp = self._create_ocp_solver_spatial()
        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    # ----------------------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------------------
    def _create_spatial_model(self):
        s = SX.sym('s')
        e_psi = SX.sym('e_psi')
        e_y = SX.sym('e_y')
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        v = SX.sym('v')
        X = vertcat(e_psi, e_y, x, y, psi, v)

        a = SX.sym('a')
        delta = SX.sym('delta')
        U = vertcat(a, delta)

        X_dot = SX.sym('xdot', X.shape[0])

        beta = arctan(self.lr * tan(delta) / self.L)
        vx = v * cos(psi + beta)
        vy = v * sin(psi + beta)
        dpsi = v * sin(beta) / self.lr

        sdot = (v * cos(beta) * cos(e_psi) - v * sin(beta) * sin(e_psi)) / (1 - self.kappa(s) * e_y)
        dx_ds = vx / sdot
        dy_ds = vy / sdot
        dv_ds = a / sdot
        dpsi_ds = dpsi / sdot
        d_e_psi = dpsi / sdot - self.kappa(s)
        d_e_y = (v * cos(beta) * sin(e_psi) + v * sin(beta) * cos(e_psi)) / sdot

        f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, dv_ds)
        f_impl = X_dot - f_expl

        model = AcadosModel()
        model.x = X
        model.xdot = X_dot
        model.u = U
        model.p = vertcat(s)
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.name = "SpatialKinematicBicycle_model"
        return model

    # ----------------------------------------------------------------------
    # OCP SOLVER
    # ----------------------------------------------------------------------
    def _create_ocp_solver_spatial(self) -> AcadosOcp:
        ocp = AcadosOcp()
        model = self._create_spatial_model()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()

        # weights
        Q = np.diag([5e3, 1e2, 5e-1])
        R = np.diag([7e-2, 1e-2])

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.model.cost_y_expr = vertcat(model.x[0] + arctan(self.lr * tan(model.u[1]) / self.L),
                                        model.x[1], model.x[-1], model.u)
        ocp.cost.yref = np.zeros((3 + nu,))

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag([5e3, 1e2]) * self.ds
        ocp.model.cost_y_expr_e = vertcat(model.x[:2])
        ocp.cost.yref_e = np.zeros(2)

        # bounds
        ocp.constraints.lbu = np.array([-2, -np.deg2rad(28)])
        ocp.constraints.ubu = np.array([2, np.deg2rad(28)])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.lbx = np.array([-np.deg2rad(20), -0.20])
        ocp.constraints.ubx = np.array([np.deg2rad(20), 0.20])
        ocp.constraints.idxbx = np.array([0, 1])

        ocp.constraints.x0 = self.X0
        ocp.parameter_values = np.array([self.s_ref[0]])

        # solver options
        ocp.solver_options.N_horizon = self.N_horizon
        ocp.solver_options.tf = self.Tf
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"

        return ocp

    # ----------------------------------------------------------------------
    # SUPPORT FUNCTIONS
    # ----------------------------------------------------------------------
    def set_initial_state(self, x0):
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

    def set_reference(self, idx):
        for j in range(self.N_horizon):
            self.acados_solver.set(j, "yref",
                np.concatenate([self.trajectory[:2, idx + j],
                                [self.trajectory[-1, idx + j]], np.zeros(2)]))
            self.acados_solver.set(j, "p", np.array([self.s_ref[idx + j]]))
        self.acados_solver.set(self.N_horizon, "yref", self.trajectory[:2, idx + self.N_horizon])
        self.acados_solver.set(self.N_horizon, "p", np.array([self.s_ref[idx + self.N_horizon]]))

    # ----------------------------------------------------------------------
    # MPC SOLVE
    # ----------------------------------------------------------------------
    def solve(self, state, idx, a_prev=None, delta_prev=None):
        self.set_initial_state(state)
        self.set_reference(idx)

        # warm start
        if a_prev is not None and delta_prev is not None:
            for j in range(self.N_horizon):
                self.acados_solver.set(j, "u", np.array([a_prev, delta_prev]))

        #print(f"[DEBUG] state_ocp: {state}")
        #print(f"[DEBUG] idx={idx}, ref_s={self.s_ref[idx]:.2f}")
        #print(f"[DEBUG] ref_xy=({self.trajectory[2, idx]:.2f},{self.trajectory[3, idx]:.2f}), ref_v={self.trajectory[5, idx]:.2f}")


        status = self.acados_solver.solve()
        u0 = self.acados_solver.get(0, "u")

        #print(f"[DEBUG] u0 = a={u0[0]:.3f}, delta={np.rad2deg(u0[1]):.2f}°")


        if status not in [0, 2]:
            raise RuntimeError(f"acados OCP solve failed with status {status}")


        return u0[0], u0[1]

    # ----------------------------------------------------------------------
    # STATE TRANSFORMATION
    # ----------------------------------------------------------------------
    def get_state(self, x, y, yaw, v):
        """
        Compute spatial state: [e_psi, e_y, x, y, psi, v]
        using the reference trajectory and current pose.
        """
        s_sim, e_psi, e_y, idx = time2spatial(x, y, yaw, self.s_ref, self.trajectory[2:5, :])
        state_ocp = np.array([e_psi, e_y, x, y, yaw, v])
        return state_ocp, idx
