
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, cos, sin, tan, arctan, interpolant, fabs
from reference_generation2dyn import TrajectoryGeneration


class MPC_DynamicBicycle:
    def __init__(self, ds, N_horizon=50, nodes=[73, 91, 125]): # [459,418]
        # vehicle parameters
        self.lf = 0.13  # distance from CoG to front wheels
        self.lr = 0.13  # distance from CoG to rear wheels
        self.L = self.lf + self.lr  # wheelbase
        self.m = 1.415
        self.I_z = 0.17423
        self.Bcf, self.Ccf, self.Dcf =  0.4, 1.3, 6.94
        self.Bcr, self.Ccr, self.Dcr = 0.4, 1.3, 6.94

        self.g = 9.81
        self.mi = 0.9

        #air charachteristics
        self.ro = 1.225
        self.Cz = -0.4
        self.Az=  0.5


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
        ocp = self.CreateOcpSolver_SpatialDyn()
        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")


    # ------------------ MODEL ------------------
    def SpatialDynamicBicycle(self)-> AcadosOcp:
        model_name = "SpatialDynamicBicycle"

        s = SX.sym('s')
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        vx  = SX.sym('vx')
        vy  = SX.sym('vy')
        e_psi = SX.sym('e_psi')
        e_y = SX.sym('e_y')
        omega  = SX.sym('r')
        x = vertcat(e_psi, e_y, x, y, vx, vy, psi, omega)

        # Controls: steering angle delta, acceleration a
        a= SX.sym("a")
        delta = SX.sym("delta")
        u = vertcat(a, delta)

        # xdot
        s_dot = SX.sym("s_dot")
        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        vx_dot = SX.sym("vx_dot")
        vy_dot = SX.sym("vy_dot")
        psi_dot = SX.sym("psi_dot")
        omega_dot   = SX.sym('omega_dot')
        e_psi_dot = SX.sym('e_psi_dot')
        e_y_dot = SX.sym('e_y_dot')
        xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot,vx_dot, vy_dot, psi_dot,omega_dot)

        beta = arctan(vy/(vx+1e-5)) # slipping angle
        # Slip angles  -  aprox of small angles
        beta_f = - arctan((vy + self.lf*omega)/(vx+1e-5)) + delta #+np.pi/2
        beta_r = arctan((vy - self.lr*omega)/(vx+1e-5))
        
        Fx_d = 1/2*self.ro* self.Cz* self.Az* vx*fabs(vx)
        # Lateral tire forces - simplified Pacejka # forget about them, try 
        Fc_f = -self.mi *self.Dcf * sin( self.Ccf * arctan(1/self.mi *self.Bcf * beta_f) )
        Fc_r = - self.mi * self.Dcr * sin( self.Ccr * arctan(1/self.mi * self.Bcr * beta_r) )

        Fyf = Fc_f*cos(delta)
        Fyr = Fc_r
        
        # DYNAMICS
        dX   = vx*cos(psi) - vy*sin(psi)
        dY  = vx*sin(psi) + vy*cos(psi)
        dvx   = vy*omega + a +  (-Fx_d - Fc_f*sin(delta))/self.m# we consider that this is completly longitudial acceleration
        dvy   = - vx*omega+ (Fyf + Fyr)/self.m
        dpsi = omega
        domega    = (self.lf*Fyf - self.lr*Fyr)/self.I_z


        #Spatial dynamics dx/ds = f(x,u)
        sdot = (vx * cos(e_psi) - vy *sin(e_psi))/(1 - self.kappa(s)* e_y)
        
        dx_ds    = dX / (sdot)
        dy_ds    = dY / (sdot)
        dvx_ds    = dvx/ (sdot)
        dvy_ds    = dvy/ (sdot)
        dpsi_ds  = (dpsi) / (sdot)
        domega_ds  = (domega) / (sdot)
        d_e_psi = (dpsi)/(sdot) - self.kappa(s)
        d_e_y = (vx  * sin(e_psi) + vy* cos(e_psi)) / (sdot)
        f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds,dvx_ds, dvy_ds, dpsi_ds, domega_ds)
        f_impl = xdot - f_expl

        # algebraic variables
        z = vertcat([])
        # parameters
        p = vertcat(s)

        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u 
        model.p = p
        model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$",  "$v_x$",  "$v_y$", "$\\psi$", "$omega$"]
        model.u_labels = [ "$a$", "$delta$"]
        model.p_labels    = ["$s$"] 
        model.name = model_name
        return model

    # ------------------ OCP SOLVER ------------------
  
    def CreateOcpSolver_SpatialDyn(self) -> AcadosOcp:
        ocp = AcadosOcp()
        model = self.SpatialDynamicBicycle()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 3 + nu
        ny_e = 3

        ocp.solver_options.N_horizon = self.N_horizon
        Q_mat = np.diag([1e2,5*1e1,1e-1])  
        R_mat =  np.diag([1e-1,1e-1])

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1],model.x[4], model.u) #
    
        #terminal costs
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        Q_mat = np.diag([1e2,5*1e1])
        ocp.cost.W_e = Q_mat*self.ds
        yref_e = np.array([0.0, 0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1]) 
    
        ocp.parameter_values  = np.array([self.s_ref[0]])
        # set constraints on the input                                                                                                             
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        # constraints on the states
        ocp.constraints.lbx = np.array([ -np.deg2rad(45), -0.4])
        ocp.constraints.ubx = np.array([ np.deg2rad(45), 0.4])
        ocp.constraints.idxbx = np.array([0,1])

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = self.Tf
        return ocp


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
            self.acados_solver.set(j, "yref", np.concatenate([self.trajectory[:2, i+j], [self.trajectory[4, i+j]], np.array([0.0, 0.0])]))
            self.acados_solver.set(j, "p", np.array([self.s_ref[i + j]]))
        self.acados_solver.set(self.N_horizon, "yref", self.trajectory[:2, i+self.N_horizon])
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
