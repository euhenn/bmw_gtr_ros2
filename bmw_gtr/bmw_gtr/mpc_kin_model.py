
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, cos, sin, tan, arctan, interpolant
from reference_generation2 import TrajectoryGeneration


class MPC_KinematicBicycle:
    def __init__(self, ds, N_horizon=50, nodes=[73, 91, 125]): # [459,418]
        # vehicle parameters
        self.lf = 0.13  # distance from CoG to front wheels
        self.lr = 0.13  # distance from CoG to rear wheels
        self.L = self.lf + self.lr  # wheelbase

        # MPC parameters
        self.ds = ds
        self.N_horizon = N_horizon
        self.Tf = self.N_horizon * self.ds
        

        # reference trajectory
        real_track = TrajectoryGeneration(self.ds, self.N_horizon)
        self.trajectory, self.s_ref, kappa_ref = real_track.generating_spatial_reference(nodes)
        self.kappa = interpolant("kappa", "bspline", [self.s_ref], kappa_ref)

        self.X0 = self.trajectory[:, 0]

        #model = self.Spatial_KinematiBicycleModel()
        ocp = self.CreateOcpSolver_SpatialKin()
        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")


    # ------------------ MODEL ------------------
    def Spatial_KinematicBicycleModel(self):
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

        beta = arctan(self.lr * tan(delta) / self.L)
        vx = v* cos(psi+beta)
        vy = v* sin(psi+beta)
        dpsi = v *sin(beta) / self.lr
        
        #Spatial dynamics dx/ds = f(x,u)
        sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta) *sin(e_psi))/(1 - self.kappa(s)* e_y)
        dx_ds    = vx / (sdot)
        dy_ds    = vy / (sdot)
        dv_ds    = a / (sdot)
        dpsi_ds  = (dpsi) / (sdot)
        d_e_psi = (dpsi)/(sdot) - self.kappa(s)
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
        model = self.Spatial_KinematicBicycleModel()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 3 + nu
        ny_e = 3

        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = np.diag([5*1e3,1e2,1e1])  
        R_mat =  np.diag([1e-2,1e-3])

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(self.lr*tan(model.u[1])/self.L),model.x[1], model.x[-1], model.u) #

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag([5*1e3,1e2, 1e1]) * self.ds
        yref_e = np.array([0.0, 0.0, 0.25]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[:2], model.x[-1]) 
    
        ocp.parameter_values  = np.array([self.s_ref[0]])
                                                                                                           
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(28)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(28)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        ocp.constraints.lbx = np.array([ -np.deg2rad(30), -0.30])
        ocp.constraints.ubx = np.array([ np.deg2rad(30), 0.30])
        ocp.constraints.idxbx = np.array([0,1])

        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = self.Tf
        return ocp

    def ClosestPoint(self, state):
        x, y, yaw, _ = state
        dist_sq = (self.trajectory[2,:] - x)**2 + (self.trajectory[3,:] - y)**2
        idxmin = np.argmin(dist_sq)
        min_dist = np.sqrt(dist_sq[idxmin])
        _, _, x0,y0, psi0, v0 = self.trajectory[:,idxmin]
        epsi=psi0 - psi
        ey=np.cos(psi0)*(y-y0)-np.sin(psi0)*(x-x0)
        full_state = np.hstack((epsi,ey,state))
        
        return self.s_ref[idxmin], full_state, idxmin, min_dist

    def set_initial_state(self, x0):
        """Set x0 constraint at stage 0."""
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)
    
    def initialize(self):
        for stage in range(self.N_horizon + 1):
            acados_solver.set(stage, "x", self.X0)
        for stage in range(self.N_horizon):
            acados_solver.set(stage, "u", np.array([0.0, 0.0]))

    def set_reference(self, i):
        if i + self.N_horizon >= len(self.s_ref):
            print("Index out of range risk!")
        for j in range(self.N_horizon):
            self.acados_solver.set(j, "yref", np.concatenate([self.trajectory[:2, i+j], [self.trajectory[-1, i+j]], np.array([0.0, 0.0])]))
            self.acados_solver.set(j, "p", np.array([self.s_ref[i + j]]))
        self.acados_solver.set(self.N_horizon, "yref", np.concatenate([self.trajectory[:2, i+self.N_horizon],[self.trajectory[-1, i+self.N_horizon]] ]))
        self.acados_solver.set(self.N_horizon, "p", np.array([self.s_ref[i+ self.N_horizon]]))

    def solve(self, state, idx, a=None, delta = None):
        # Debug
        #print("DEBUG solve: state:", state)
        #print("DEBUG solve: traj shape:", traj.shape)

        self.set_initial_state(state)
        self.set_reference(idx)

        # show first few yrefs
        try:
            print("DEBUG yref[0]:", self.acados_solver.get(0, "yref"))
            print("DEBUG yref[1]:", self.acados_solver.get(1, "yref"))
        except Exception:
            pass

        # simple warm start: repeat previous u (or zeros) across horizon
        if (a is not None) and (delta is not None):
            for j in range(self.N_horizon):
                try:
                    self.acados_solver.set(j, "u", np.array([a, delta]))
                except Exception:
                    pass
        
                
        status = self.acados_solver.solve()
        #print("DEBUG acados status:", status)
        u0 = self.acados_solver.get(0, "u")
        #print("DEBUG u0:", u0)
        if status not in [0, 2]:
            raise RuntimeError(f"acados OCP solve failed with status {status}")

        return u0[0], u0[1]


       # ------------------ CLOSED LOOP SIMULATION ------------------
    

if __name__ == "__main__":
    mpc = MPC_KinematicBicycle()
    x0 = np.array([0.0, 0.0, 0.406, 6.56, -1.57, 1])
    
    # Debug: Check trajectory shape
    print(f"Trajectory shape: {mpc.trajectory.shape}")
    print(f"Trajectory shape: {mpc.trajectory[:,0]}")
    
    a_cmd, delta_cmd = mpc.solve(x0, 1)
    print("First command:", a_cmd, delta_cmd)
