
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, cos, sin, tan, arctan
from reference_trajectory_generation import TrajectoryGeneration


class MPC_KinematicBicycle:
    def __init__(self, ds=0.1, N_horizon=20, nodes=[73, 91]):
        # vehicle parameters
        self.lf = 0.13  # distance from CoG to front wheels
        self.lr = 0.13  # distance from CoG to rear wheels
        self.L = self.lf + self.lr  # wheelbase

        # MPC parameters
        self.ds = ds
        self.N_horizon = N_horizon
        self.Tf = N_horizon * ds
        self.dt_sim = dt_sim

        # reference trajectory
        real_track = TrajectoryGeneration()
        self.trajectory, self.N = real_track.spatial_reference(N_horizon, nodes)
        self.X0 = self.trajectory[:, 0]

        model = self.Time_KinematicBicycleModel()
        ocp = self.CreateOcpSolver_TimeKin()
        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")


    # ------------------ MODEL ------------------
    def Spatial_KinematiBicycleModel(self):
        s = SX.sym('s')
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        v= SX.sym("v")
        e_psi = SX.sym('e_psi')
        e_y = SX.sym('e_y')
        x = vertcat(e_psi, e_y, x, y, psi, v)

        a= SX.sym("a")
        delta = SX.sym("delta")
        u = vertcat(a, delta)

        s_dot = SX.sym("s_dot")
        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        psi_dot = SX.sym("psi_dot")
        v_dot = SX.sym("v_dot")
        e_psi_dot = SX.sym('e_psi_dot')
        e_y_dot = SX.sym('e_y_dot')
        xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,v_dot)

        beta = arctan(lr * tan(delta) / L)
        vx = v* cos(psi+beta)
        vy = v* sin(psi+beta)
        dpsi = v *sin(beta) / lr
        
        #Spatial dynamics dx/ds = f(x,u)
        sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta) *sin(e_psi))/(1 - kappa(s)* e_y)
        dx_ds    = vx / (sdot)
        dy_ds    = vy / (sdot)
        dv_ds    = a / (sdot)
        dpsi_ds  = (dpsi) / (sdot)
        d_e_psi = (dpsi)/(sdot) - kappa(s)
        d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) / (sdot)
        f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, dv_ds)
        f_impl = xdot - f_expl

        z = vertcat([])
        p = vertcat(s)

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u 
        model.p = p
        model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$v$"]
        model.u_labels = [ "$a$", "$delta$"]
        model.p_labels    = ["$s$"] 
        model.name = "SpatialKinematicBicycle_model"
        return model

    # ------------------ OCP SOLVER ------------------
    def CreateOcpSolver_SpatialKin(self) -> AcadosOcp:
        ocp = AcadosOcp()
        model = Spatial_KinematiBicycleModel()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 3 + nu
        ny_e = 3

        ocp.solver_options.N_horizon = N_horizon
        Q_mat = np.diag([1e2,5*1e1,1e1])  
        R_mat =  np.diag([1e0,1e0])

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.x[-1], model.u) #

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = Q_mat*ds
        yref_e = np.array([0.0, 0.0,0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[:2], model.x[-1]) 
    
        ocp.parameter_values  = np.array([s_ref[0]])
                                                                                                           
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(60)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(60)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = X0

        ocp.constraints.lbx = np.array([ -np.deg2rad(20), -0.5])
        ocp.constraints.ubx = np.array([ np.deg2rad(20), 0.5])
        ocp.constraints.idxbx = np.array([0,1])

        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = Tf
        return ocp



    def set_initial_state(self, x0):
        """Set x0 constraint at stage 0."""
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)
    
    def initialize(self):
        for stage in range(N_horizon + 1):
            acados_ocp_solver.set(stage, "x", self.X0)
        for stage in range(N_horizon):
            acados_ocp_solver.set(stage, "u", np.array([0.0, 0.0]))

    def set_reference(self, i):
        for j in range(self.N_horizon):
            self.acados_solver.set(j, "yref", np.concatenate([self.trajectory[:, i+j], np.array([1.0, 0.0])]))
        self.acados_solver.set(N_horizon, "yref", self.trajectory[:2, i+N_horizon])

    def solve(self, state, idx):
        # Debug
        #print("DEBUG solve: state:", state)
        #print("DEBUG solve: traj shape:", traj.shape)

        self.set_initial_state(state)
        self.set_reference(idx)

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


       # ------------------ CLOSED LOOP SIMULATION ------------------
    

if __name__ == "__main__":
    mpc = MPC_KinematicBicycle()
    x0 = np.array([0.406, 6.56, -1.57])
    
    # Debug: Check trajectory shape
    print(f"Trajectory shape: {mpc.trajectory.shape}")
    
    traj = mpc.get_reference_segment(1)
    print(f"Reference segment shape: {traj.shape}")
    
    v_cmd, delta_cmd = mpc.solve(x0, traj)
    print("First command:", v_cmd, delta_cmd)
