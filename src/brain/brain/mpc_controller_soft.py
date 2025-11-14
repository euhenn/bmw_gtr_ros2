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
    def __init__(self, ds=0.05, N_horizon=50, nodes=[73, 97, 125, 150,135]):
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
        self.traj_gen = TrajectoryGeneration(self.ds, self.N_horizon,use_curvature_velocity=False, v_max=0.5, v_min=0.4, smooth_velocity=True)
        self.traj, self.s_ref, kappa_ref = self.traj_gen.generating_spatial_reference(nodes)
        self.kappa = interpolant("kappa", "bspline", [self.s_ref], kappa_ref)
        self.x0 = self.traj[:, 0]

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------
    def _make_model(self):
        s = SX.sym("s")
        epsi, ey, x, y, psi, v = SX.sym("e_psi"), SX.sym("e_y"), SX.sym("x"), SX.sym("y"), SX.sym("psi"), SX.sym("v")
        a, delta = SX.sym("a"), SX.sym("delta")

        X = vertcat(epsi, ey, x, y, psi, v)
        U = vertcat(a, delta)
        Xdot = SX.sym("Xdot", X.shape[0])

        beta = arctan(self.lr * tan(delta) / self.L)
        vx, vy = v * cos(psi + beta), v * sin(psi + beta)
        dpsi = v * sin(beta) / self.lr
        #sdot = (v * cos(beta) * cos(epsi) - v * sin(beta) * sin(epsi)) / (1 - self.kappa(s) * ey)
        eps = 5e-3
        sdot = (v * cos(beta) * cos(epsi) - v * sin(beta) * sin(epsi))
        sdot = sdot / (1 - self.kappa(s) * ey + eps)


        f_expl = vertcat(
            dpsi / sdot - self.kappa(s),
            (v * cos(beta) * sin(epsi) + v * sin(beta) * cos(epsi)) / sdot,
            vx / sdot,
            vy / sdot,
            dpsi / sdot,
            a / sdot,
        )

        model = AcadosModel()
        model.x, model.xdot, model.u, model.p = X, Xdot, U, vertcat(s)
        model.f_expl_expr, model.f_impl_expr = f_expl, Xdot - f_expl
        model.name = "SpatialBicycle"
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
        # --- your Q, R tuning ---
        Q = np.diag([5e3, 1e2, 1e2])   
        R = np.diag([5e-10, 1e2])  

        ocp.cost.W_e = np.diag([5e3, 1e2]) * self.ds
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        m = ocp.model
        ocp.model.cost_y_expr = vertcat(
            m.x[0] + arctan(self.lr * tan(m.u[1]) / self.L),  # e_psi + beta
            m.x[1],                                          # e_y
            m.x[-1],                                         # v
            m.u,                                             # a, delta
        )
        ocp.model.cost_y_expr_e = vertcat(m.x[0], m.x[1])
        ocp.cost.yref = np.zeros(3 + nu)
        ocp.cost.yref_e = np.zeros(2)

        # ---------------------------------------------------
        # SOFT CONSTRAINT PENALTY for e_psi (x[0]) and e_y (x[1])
        # ---------------------------------------------------
        # We are softening 2 state bounds (e_psi and e_y)
        nsbx = 2

        # Penalty on slack (bigger = harder constraint)
        w_soft_epsi = 1e4   # heading error violation
        w_soft_ey   = 1e3   # lateral error violation

        ocp.cost.Zl = np.diag([w_soft_epsi, w_soft_ey])
        ocp.cost.Zu = np.diag([w_soft_epsi, w_soft_ey])
        ocp.cost.zl = np.zeros(nsbx)
        ocp.cost.zu = np.zeros(nsbx)
    
    def _configure_constraints(self, ocp, nx, nu):
        # -------- Inputs (hard) --------
        ocp.constraints.lbu = np.array([-20, -np.deg2rad(28)])
        ocp.constraints.ubu = np.array([ 20,  np.deg2rad(28)])
        ocp.constraints.idxbu = np.arange(nu)

        # -------- States (e_psi, e_y) with SOFT bounds --------
        # state order: [e_psi, e_y, x, y, psi, v]
        ocp.constraints.lbx   = np.array([-np.deg2rad(45), -0.35])
        ocp.constraints.ubx   = np.array([ np.deg2rad(45),  0.35])
        ocp.constraints.idxbx = np.array([0, 1])  # indices of e_psi and e_y

        # Tell acados that these 2 state bounds have slacks
        nsbx = ocp.constraints.idxbx.size   # = 2
        ocp.constraints.lsbx   = np.zeros(nsbx)   # ideal slack = 0
        ocp.constraints.usbx   = np.zeros(nsbx)
        ocp.constraints.idxsbx = np.arange(nsbx)  # soften all bx above

        # If you want to LIMIT how much they can violate, you could instead do:
        max_violation = np.array([np.deg2rad(35), 0.15])  # extra allowed
        ocp.constraints.lsbx = -max_violation
        ocp.constraints.usbx =  max_violation

        # -------- Initial condition --------
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


        # ocp.solver_options.qp_solver_iter_max = 50
        # ocp.solver_options.nlp_solver_max_iter = 20
        # ocp.solver_options.levenberg_marquardt = 1e-6

        # --- Integration method accuracy ---
        #ocp.solver_options.sim_method_num_stages = 4 #50 nodes × 3 steps × 4 stages = 600 evaluations per solve!
        #ocp.solver_options.sim_method_num_steps = 3

        # # --- Tolerances ---
        # ocp.solver_options.qp_solver_tol_stat = 1e-2
        # ocp.solver_options.qp_solver_tol_eq = 1e-2
        # ocp.solver_options.qp_solver_tol_ineq = 1e-2
        # ocp.solver_options.qp_solver_tol_comp = 1e-2

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

    def get_state(self, x: float, y: float, yaw: float, v: float) -> tuple:
        """
        Compute spatial state using the reference trajectory and current pose.
        """
        _, e_psi, e_y, idx = self.traj_gen.time2space(x, y, yaw)
        state_ocp = np.array([e_psi, e_y, x, y, yaw, v])

        return state_ocp, idx
