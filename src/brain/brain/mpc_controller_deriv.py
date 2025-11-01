"""
| Parameter                | Meaning                               | Effect of ↑ increase                               | Effect of ↓ decrease                              |
| ------------------------ | ------------------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| `Q[0,0]` (heading error) | Penalizes yaw misalignment            | Vehicle turns more aggressively to correct heading | Slower heading corrections, smoother              |
| `Q[1,1]` (lateral error) | Penalizes offset from path center     | Hug path tightly, but risk oscillations            | Allows small drift but smoother path              |
| `Q[2,2]` (speed error)   | Penalizes deviation from target speed | Accelerates/decelerates strongly                   | Sluggish speed tracking                           |
| `R[0,0]` (accel)         | Penalizes strong acceleration         | Smoother throttle, slower to reach v_ref           | Reacts faster but more jerky                      |
| `R[1,1]` (steering)      | Penalizes steering magnitude          | Smoother steering, larger steady-state errors      | Aggressive steering, possibly oscillatory         |
| `W_e`                    | Penalizes final error at horizon      | Forces long-term convergence                       | More freedom short-term, less stability long-term |

"""


#!/usr/bin/env python3
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import scipy.linalg
from casadi import SX, vertcat, cos, sin, tan, arctan, interpolant
from reference_generation_velocity import TrajectoryGeneration


class MPC_KinematicBicycle:
    def __init__(self, ds=0.05, N_horizon=50, nodes=[73,97,125,150]):
        self.lf, self.lr = 0.1335, 0.1335
        self.L = self.lf + self.lr
        self.ds, self.N_horizon = ds, N_horizon
        self.Tf = N_horizon * ds
        
        self._load_reference(nodes)
        self._build_solver()

    # ----------------------------------------------------------
    # Reference trajectory
    # ----------------------------------------------------------
    def _load_reference(self, nodes):
        self.traj_gen = TrajectoryGeneration(self.ds, self.N_horizon,use_curvature_velocity=False, v_max=0.5, v_min=0.2, smooth_velocity=True)
        self.traj, self.s_ref, kappa_ref = self.traj_gen.generating_spatial_reference(nodes)
        self.kappa = interpolant("kappa", "bspline", [self.s_ref], kappa_ref)
        self.x0 =np.concatenate((self.traj[:, 0], [0.0]))


    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------
    def _make_model(self):
        s = SX.sym("s")
        epsi, ey, x, y, psi, v, delta = SX.sym("e_psi"), SX.sym("e_y"), SX.sym("x"), SX.sym("y"), SX.sym("psi"), SX.sym("v"),  SX.sym("delta")
        a, ddelta = SX.sym("a"), SX.sym("ddelta")

        X = vertcat(epsi, ey, x, y, psi, v, delta)
        U = vertcat(a, ddelta)
        Xdot = SX.sym("Xdot", X.shape[0])

        beta = arctan(self.lr * tan(delta) / self.L)
        vx, vy = v * cos(psi + beta), v * sin(psi + beta)
        dpsi = v * sin(beta) / self.lr
        #sdot = (v * cos(beta) * cos(epsi) - v * sin(beta) * sin(epsi)) / (1 - self.kappa(s) * ey)
        eps = 2e-3
        sdot = (v * cos(beta) * cos(epsi) - v * sin(beta) * sin(epsi))
        sdot = sdot / (1 - self.kappa(s) * ey + eps)


        f_expl = vertcat(
            dpsi / sdot - self.kappa(s),
            (v * cos(beta) * sin(epsi) + v * sin(beta) * cos(epsi)) / sdot,
            vx / sdot,
            vy / sdot,
            dpsi / sdot,
            a / sdot,
            ddelta/sdot,
        )

        model = AcadosModel()
        model.x, model.xdot, model.u, model.p = X, Xdot, U, vertcat(s)
        model.f_expl_expr, model.f_impl_expr = f_expl, Xdot - f_expl
        model.name = "SpatialBicycleDDelta"
        return model

    # ----------------------------------------------------------
    # OCP configuration
    # ----------------------------------------------------------
    def _build_solver(self):
        ocp = AcadosOcp()
        ocp.model = self._make_model()
        nx, nu = ocp.model.x.size()[0], ocp.model.u.size()[0]

        self._configure_costs(ocp, nx, nu)
        self._configure_constraints(ocp, nx, nu)
        self._configure_solver_options(ocp)

        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        #self.solver = AcadosOcpSolver(None,generate=False,build=False,json_file="acados_ocp.json")


    def _configure_costs(self, ocp, nx, nu):

        #DON T MODIFY
        Q = np.diag([5e3, 1e2, 5e0])
        R = np.diag([2e-2, 1e2])
        #DON T MODIFY (VIDEO 2025.10.25 used at 11:12)
        Q = np.diag([5e3, 1e2, 5e-1])
        R = np.diag([2e-2, 1e2])

        Q = np.diag([5e3, 1e2, 1e2])
        R = np.diag([5e-3, 5e-3])

        ocp.cost.W_e = np.diag([5e3, 1e2]) * self.ds # normalizing the terminal cost so its contribution stays roughly proportional to one spatial step’s cost
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        m = ocp.model
        ocp.model.cost_y_expr = vertcat(
            m.x[0] + arctan(self.lr * tan(m.x[6]) / self.L),
            m.x[1],
            m.x[5],
            m.u,
        )
        ocp.model.cost_y_expr_e = vertcat(m.x[0], m.x[1])
        ocp.cost.yref = np.zeros(3 + nu)
        ocp.cost.yref_e = np.zeros(2)

    def _configure_constraints(self, ocp, nx, nu):
        ocp.constraints.lbu = np.array([-5.0, -4.0])
        ocp.constraints.ubu = np.array([5.0, 4.0])
        ocp.constraints.idxbu = np.arange(nu)

        #ocp.constraints.lbx = np.array([-np.deg2rad(130), -1.5])
        #ocp.constraints.ubx = np.array([np.deg2rad(130), 1.5])
        ocp.constraints.lbx = np.array([-np.deg2rad(50), -0.45, -np.deg2rad(28)])
        ocp.constraints.ubx = np.array([np.deg2rad(50), 0.45,  np.deg2rad(28)])
        ocp.constraints.idxbx = np.array([0, 1, 6])

        ocp.constraints.x0 = self.x0
        ocp.parameter_values = np.array([self.s_ref[0]])

    def _configure_solver_options(self, ocp):
        ocp.solver_options.N_horizon = self.N_horizon
        ocp.solver_options.tf = self.Tf

        # --- Robust QP solver ---
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK" # (‘ERK’, ‘IRK’, ‘GNSF’, ‘DISCRETE’, ‘LIFTED_IRK’)
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # or "SQP" for full iterations

        # --- Stability & convergence tuning ---
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"


    # ----------------------------------------------------------
    # Reference and solving
    # ----------------------------------------------------------
    def update_reference_window(self, idx):
        for j in range(self.N_horizon):
            ey_ref, epsi_ref = self.traj[:2, idx + j]
            v_ref = self.traj[-1, idx + j]
            yref = np.array([epsi_ref, ey_ref, v_ref, 0, 0])
            self.solver.set(j, "yref", yref)
            self.solver.set(j, "p", np.array([self.s_ref[idx + j]]))

        self.solver.set(self.N_horizon, "yref", self.traj[:2, idx + self.N_horizon])
        self.solver.set(self.N_horizon, "p", np.array([self.s_ref[idx + self.N_horizon]]))


    def set_initial_state(self, x0):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

    def solve(self, x0, idx, warm_start=None):
        self.set_initial_state(x0)
        self.update_reference_window(idx)

        if warm_start is not None:
            for j in range(self.N_horizon):
                self.solver.set(j, "u", warm_start)

        status = self.solver.solve()
        if status not in [0, 2]:
            raise RuntimeError(f"Solver failed with status {status}")

        return self.solver.get(0, "u")



    # ----------------------------------------------------------------------
    # STATE TRANSFORMATION
    # ----------------------------------------------------------------------

    def get_state(self, x: float, y: float, yaw: float, v: float, delta: float) -> tuple:
        """
        Compute spatial state using the reference trajectory and current pose.
        """
        _, e_psi, e_y, idx = self.traj_gen.time2space(x, y, yaw)
        state_ocp = np.array([e_psi, e_y, x, y, yaw, v, delta])

        return state_ocp, idx