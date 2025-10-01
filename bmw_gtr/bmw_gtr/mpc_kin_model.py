from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, cos, sin, tan, arctan
from reference_trajectory_generation import TrajectoryGeneration


class MPC_KinematicBicycle:
    def __init__(self, dt_ocp=0.1, N_horizon=20, dt_sim=0.1, nodes=[73, 91]):
        # vehicle parameters
        self.lf = 0.13  # distance from CoG to front wheels
        self.lr = 0.13  # distance from CoG to rear wheels
        self.L = self.lf + self.lr  # wheelbase

        # MPC parameters
        self.dt_ocp = dt_ocp
        self.N_horizon = 20 
        self.Tf = N_horizon * dt_ocp
        self.dt_sim = dt_sim

        # reference trajectory
        real_track = TrajectoryGeneration()
        self.trajectory, self.N = real_track.time_reference(N_horizon, nodes)
        self.X0 = self.trajectory[:, 0]

        self.T = int(self.N * self.dt_ocp)
        self.Nsim = int(self.T / self.dt_sim)

        self.model = self.Time_KinematicBicycleModel()
        self.ocp = self.CreateOcpSolver_TimeKin()
        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_time_kin_nonlinear.json")

        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()


    # ------------------ MODEL ------------------
    def Time_KinematicBicycleModel(self):
        x = SX.sym("x")
        y = SX.sym("y")
        psi = SX.sym("psi")
        x_var = vertcat(x, y, psi)

        v = SX.sym("v")
        delta = SX.sym("delta")
        u = vertcat(v, delta)

        # derivatives
        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        psi_dot = SX.sym("psi_dot")
        xdot = vertcat(x_dot, y_dot, psi_dot)

        # dynamics
        beta = arctan(self.lr * tan(delta) / self.L)  # slip angle
        f_expl = vertcat(v * cos(psi + beta), v * sin(psi + beta), v * sin(beta) / self.lr)
        f_impl = xdot - f_expl

        model = AcadosModel()
        model.name = "time_kin_bicycle"
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x_var
        model.xdot = xdot
        model.u = u

        return model

    def Spatial_KinematiBicycleModel(self):
        s = SX.sym('s')
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        v= SX.sym("v")
        e_psi = SX.sym('e_psi')
        e_y = SX.sym('e_y')
        x = vertcat(e_psi, e_y, x, y, psi, v)

        # Controls: steering angle delta, acceleration a
        a= SX.sym("a")
        delta = SX.sym("delta")
        u = vertcat(a, delta)

        # xdot
        s_dot = SX.sym("s_dot")
        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        psi_dot = SX.sym("psi_dot")
        v_dot = SX.sym("v_dot")
        e_psi_dot = SX.sym('e_psi_dot')
        e_y_dot = SX.sym('e_y_dot')
        xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,v_dot)

        beta = arctan(lr * tan(delta) / L) # slipping angle
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
        model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$v$"]
        model.u_labels = [ "$a$", "$delta$"]
        model.p_labels    = ["$s$"] 
        model.name = "SpatialKinematicBicycle_model"
        return model

    # ------------------ OCP SOLVER ------------------
    def CreateOcpSolver_TimeKin(self):
        ocp = AcadosOcp()
        model = self.Time_KinematicBicycleModel()
        ocp.model = model

        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx

        ocp.solver_options.N_horizon = self.N_horizon

        # cost weights
        Q_mat = 2 * np.diag([1e2, 1e2, 1e2])
        R_mat = 2 * np.diag([1e0, 1e1])

        # path cost
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[:2],
                                        model.x[2]+ arctan(self.lr * tan(model.u[1]) / self.L),
                                        model.u)

        # terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = 2 * np.diag([1e2, 1e2]) * self.dt_ocp
        ocp.cost.yref_e = np.array([4.0, 1.0])
        ocp.model.cost_y_expr_e = vertcat(model.x[:2])

        # constraints
        ocp.constraints.lbu = np.array([-3, -np.deg2rad(30)])
        ocp.constraints.ubu = np.array([3, np.deg2rad(30)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.X0

        # solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.tf = self.Tf

        return ocp

    def CreateOcpSolver_SpatialKin(self) -> AcadosOcp:
        ocp = AcadosOcp()
        model = Spatial_KinematiBicycleModel()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = 3 + nu
        ny_e = 3

        ocp.solver_options.N_horizon = N_horizon
        Q_mat = np.diag([1e1,5*1e1,1e3])  
        R_mat =  np.diag([1e0,1e0])

        #path const
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((ny,))
        ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.x[-1], model.u) #
    
        #terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = Q_mat*ds_ocp
        yref_e = np.array([0.0, 0.0,0.0]) 
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = vertcat(model.x[:2], model.x[-1]) 
    
        ocp.parameter_values  = np.array([s_ref[0]])
        # set constraints on the input                                                                                                             
        ocp.constraints.lbu = np.array([-1, -np.deg2rad(60)])
        ocp.constraints.ubu = np.array([1, np.deg2rad(60)])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = X0

        # constraints on the states
        ocp.constraints.lbx = np.array([ -np.deg2rad(20), -0.5])
        ocp.constraints.ubx = np.array([ np.deg2rad(20), 0.5])
        ocp.constraints.idxbx = np.array([0,1])

        # set options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 70
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.tf = Tf
        return ocp



    def set_initial_state(self, x0):
        """Set x0 constraint at stage 0."""
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)
    
    def initialize_solver(self):
        for stage in range(N_horizon + 1):
            acados_ocp_solver.set(stage, "x", self.X0)
        for stage in range(N_horizon):
            acados_ocp_solver.set(stage, "u", np.array([0.0, 0.1]))

    def set_reference(self, i):
        """
        traj: shape (3, N_horizon+1) with rows [x_ref, y_ref, psi_ref]
        Sets yref for stages 0..N-1 and terminal yref_e at stage N.
        """
        #assert traj.shape[0] == 3
        #assert traj.shape[1] >= self.N_horizon + 1

        for j in range(self.N_horizon):
            # yref = [x_ref, y_ref, psi_ref + slip_target(=0 here), v_ref(=0), delta_ref(=0)]
            self.acados_solver.set(j, "yref", np.concatenate([self.trajectory[:, i+j], np.array([1.0, 0.0])]))
        self.acados_solver.set(N_horizon, "yref", self.trajectory[:2, i+N_horizon])


    def warm_start(self, x_guess=None, u_guess=None):
        """Optional: initialize the trajectory for faster convergence."""
        if x_guess is not None:
            for k in range(self.N_horizon + 1):
                self.acados_solver.set(k, "x", x_guess)
        if u_guess is not None:
            for k in range(self.N_horizon):
                self.acados_solver.set(k, "u", u_guess)

    def solve(self):
        status = self.acados_solver.solve()
        if status not in [0, 2]:
            raise RuntimeError(f"acados OCP solve failed with status {status}")
            #print(f"ACADOS solver failed with status {status}")
        u0 = self.acados_solver.get(0, "u")  # [v, delta]
        return u0

       # ------------------ CLOSED LOOP SIMULATION ------------------
    def ClosedLoopSimulationSIM(self):
        ocp = self.CreateOcpSolver_TimeKin()
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_nonlinear.json")

        sim = AcadosSim()
        sim.model = ocp.model
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
        sim.solver_options.T = self.dt_sim
        acados_sim_solver = AcadosSimSolver(sim, json_file="acados_sim_nonlinear.json")

        N_horizon = acados_ocp_solver.N
        nx = ocp.model.x.rows()
        nu = ocp.model.u.rows()

        simX = np.zeros((self.Nsim + 1, nx))
        simU = np.zeros((self.Nsim, nu))
        yref_ = np.zeros((self.N + N_horizon + 1, nx))

        xcurrent = self.X0
        simX[0, :] = xcurrent

        # initialize solver
        for stage in range(N_horizon + 1):
            acados_ocp_solver.set(stage, "x", xcurrent)
        for stage in range(N_horizon):
            acados_ocp_solver.set(stage, "u", np.array([0.0, 0.1]))
        for stage in range(N_horizon + self.N):
            yref_[stage, :] = self.trajectory[:, stage]

        # simulation loop
        for i in range(self.Nsim):
            k = int(self.dt_ocp / self.dt_sim)
            if i % k == 0:
                for j in range(N_horizon):
                    acados_ocp_solver.set(j, "yref", np.concatenate((self.trajectory[:, i // k + j], [0.0, 0.0])))
                acados_ocp_solver.set(N_horizon, "yref", self.trajectory[:2, i // k + N_horizon])

            status = acados_ocp_solver.solve()
            if status not in [0, 2]:
                print(f"ACADOS solver failed with status {status}")

            u0 = acados_ocp_solver.get(0, "u")
            xcurrent = acados_sim_solver.simulate(xcurrent, u0)

            simU[i, :] = u0
            simX[i + 1, :] = xcurrent
            simX[i + 1, 2] = simX[i + 1, 2] + arctan(self.lr * tan(simU[i,1]) / self.L)
            

            acados_ocp_solver.set(0, "lbx", xcurrent)
            acados_ocp_solver.set(0, "ubx", xcurrent)

        self.plot_trajectory(simX, simU, yref_)

       def set_initial_state(self, x0):
        """Set x0 constraint at stage 0."""
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)
   
    # ------------------ PLOTTING ------------------
    def plot_trajectory(self, simX, simU, yref_):
        timestampsx = np.linspace(0, (self.Nsim + 1) * self.dt_sim, self.Nsim + 1)
        timestampsu = np.linspace(0, self.Nsim * self.dt_sim, self.Nsim)
        timestampsy = np.linspace(0, (self.N ) * self.dt_ocp, self.N + 1)
        nx = simX.shape[1]
        ny = simU.shape[1]

        plt.figure()
        plt.plot(simX[:, 0], simX[:, 1], label="Simulation")
        plt.plot(yref_[: self.N + 1, 0], yref_[: self.N + 1, 1], "--", c="orange", label="Reference")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Trajectory")
        plt.axis("equal")
        plt.legend()

        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 8))
        fig.suptitle("States and Control over Time", fontsize=14, y=0.97)
        labels = ["x", "y", "heading+slip angle", "v", "steering"]

        for i in range(nx):
            ax[i].plot(timestampsx, simX[:, i])
            ax[i].plot(timestampsy, yref_[: self.N + 1, i], "--", label="Reference")
            ax[i].set_ylabel(labels[i])
 
        for i in range(ny):
            ax[i + 3].plot(timestampsu, simU[:, i])
            ax[i + 3].set_ylabel(labels[i + 3])

        ax[-1].set_xlabel("time [s]")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mpc = MPC_KinematicBicycle()
    mpc.ClosedLoopSimulationOCP()
