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
from casadi import SX, vertcat, cos, sin, tan, arctan, interpolant, fabs
from reference_generation_velocity import TrajectoryGeneration


class MPC_DynamicBicycle:
    def __init__(self, ds=0.05, N_horizon=50, nodes=[73, 97, 125, 150]):
        self.lf, self.lr = 0.1335, 0.1335
        self.L = self.lf + self.lr
        self.m = 1.415
        self.h = 0.03
        self.I_z = 0.17423
        self.Bcf, self.Ccf, self.Dcf = 0.425, 1.3, 6.246 
        self.Bcr, self.Ccr, self.Dcr = 0.425, 1.3, 6.246

        self.g = 9.81
        self.mi = 0.9

        #air charachteristics
        self.ro = 1.2
        self.Cd = 0.32
        self.Az=  0.021

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
        omega = np.gradient(self.traj[:, 4], self.ds)
        self.x0 = np.concatenate((self.traj[:, 0], [0.0, omega[0]]))

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------
    def _make_model(self):
        s = SX.sym("s")
        epsi, ey, x, y, psi,vx,vy, omega = SX.sym("e_psi"), SX.sym("e_y"), SX.sym("x"), SX.sym("y"), SX.sym("psi"), SX.sym("vx"), SX.sym("vy"), SX.sym("omega")
        a, delta = SX.sym("a"), SX.sym("delta")

        X = vertcat(epsi, ey, x, y, psi, vx, vy, omega)
        U = vertcat(a, delta)
        Xdot = SX.sym("Xdot", X.shape[0])

        eps = 5e-3

        # slip angles (approx)
        beta = arctan(self.lr * tan(delta) / self.L)
        beta_f = -arctan((vy + self.lf * omega) / (vx + eps)) + delta
        beta_r = arctan((vy - self.lr * omega) / (vx + eps))

         # longitudinal drag (simple quadratic)
        Fx_d = 0.5 * self.ro * self.Cz * self.Az * vx * fabs(vx)

        # simplified lateral (Pacejka-like) forces (scaled)
        Fc_f = - self.Dcf * sin(self.Ccf * arctan( self.Bcf * beta_f))
        Fc_r = - self.Dcr * sin(self.Ccr * arctan( self.Bcr * beta_r))

        # lateral components considering steering (approx)
        Fyf = Fc_f * cos(delta)  # small-angle approx; cos(delta) ~ 1 for small delta
        Fyr = Fc_r

        dX = vx * cos(psi) - vy * sin(psi)
        dY = vx * sin(psi) + vy * cos(psi)
        dvx = vy * omega + a + (-Fx_d - Fc_f * sin(delta)) / self.m
        dvy = -vx * omega + (Fyf + Fyr) / self.m
        dpsi = omega
        domega = (self.lf * Fyf - self.lr * Fyr) / self.I_z

        #sdot = (vx * cos(epsi) - vy * sin(epsi)) / (1 - self.kappa(s) * ey)
        sdot = (vx * cos(epsi) - vy * sin(epsi))
        sdot = sdot / (1 - self.kappa(s) * ey + eps)


        f_expl = vertcat(
            dpsi / sdot - self.kappa(s),
            (vx * sin(epsi) + vy * cos(epsi)) / sdot,
            dX / sdot,
            dY / sdot,
            dpsi / sdot,
            dvx / sdot,
            dvy / sdot,
            domega / sdot,
        )

        model = AcadosModel()
        model.x, model.xdot, model.u, model.p = X, Xdot, U, vertcat(s)
        model.f_expl_expr, model.f_impl_expr = f_expl, Xdot - f_expl
        model.name = "SpatialDynBicycle"
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

        self.solver = AcadosOcpSolver(ocp, json_file="acados_dynocp.json")
        #self.solver = AcadosOcpSolver(None,generate=False,build=False,json_file="acados_ocp.json")


    def _configure_costs(self, ocp, nx, nu):

        #DON T MODIFY
        Q = np.diag([5e3, 1e2, 5e0])   
        R = np.diag([2e-2, 1e2])      
        #DON T MODIFY (VIDEO 2025.10.25 used at 11:12)
        Q = np.diag([5e3, 1e2, 5e-1])   
        R = np.diag([2e-2, 1e2])    

        # r speed at e-5 goes crazy, crazy good
        Q = np.diag([5e3, 1e2, 5e-1])   
        R = np.diag([5e-6, 1e-2])    

        Q = np.diag([5e3, 1e2, 1e2])   
        R = np.diag([5e-10, 1e2])  
        ocp.cost.W_e = np.diag([5e3, 1e2]) * self.ds
        
        # Terminal cost (scaled by ds so it compares to one spatial step�s cost)
        #ocp.cost.W_e = np.diag([5e3, 1e2]) * self.ds

        #ocp.cost.W_e = np.diag([5e3, 1e2]) * self.ds # normalizing the terminal cost so its contribution stays roughly proportional to one spatial step’s cost
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        m = ocp.model
        ocp.model.cost_y_expr = vertcat(
            m.x[0] + arctan(m.x[6]/ (m.x[5]+1e-5)),
            m.x[1],
            m.x[5],
            m.u,
        )
        ocp.model.cost_y_expr_e = vertcat(m.x[0], m.x[1])
        ocp.cost.yref = np.zeros(3 + nu)
        ocp.cost.yref_e = np.zeros(2)
       
    def _configure_constraints(self, ocp, nx, nu):
        ocp.constraints.lbu = np.array([-20, -np.deg2rad(28)])
        ocp.constraints.ubu = np.array([20, np.deg2rad(28)])
        ocp.constraints.idxbu = np.arange(nu)

        ocp.constraints.lbx = np.array([-np.deg2rad(45), -0.35])
        ocp.constraints.ubx = np.array([np.deg2rad(45), 0.35])
        ocp.constraints.idxbx = np.array([0, 1])
 

        ocp.constraints.x0 = self.x0
        ocp.parameter_values = np.array([self.s_ref[0]])

    def _configure_solver_options(self, ocp):
        ocp.solver_options.N_horizon = self.N_horizon
        ocp.solver_options.tf = self.Tf

        # --- Robust QP solver ---
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK" 
        ocp.solver_options.nlp_solver_type = "SQP_RTI" 

        # --- Stability & convergence tuning ---
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"


    # ----------------------------------------------------------
    # Reference and solving
    # ----------------------------------------------------------
    def update_reference_window(self, idx):
        for j in range(self.N_horizon):
            epsi_ref, ey_ref= self.traj[:2, idx + j]
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

    def get_state(self, x: float, y: float, yaw: float, vx: float, vy: float, omega: float ) -> tuple:
        """
        Compute spatial state using the reference trajectory and current pose.
        """
        _, e_psi, e_y, idx = self.traj_gen.time2space(x, y, yaw)
        state_ocp = np.array([e_psi, e_y, x, y, yaw, vx, vy, omega])

        return state_ocp, idx
