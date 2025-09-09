from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan
from reference_trajectory_generation import *


# ------------------ INITIAL CONFIGURATION ------------------
dt_ocp = 0.1
N_horizon = 20
Tf =N_horizon * dt_ocp

real_track = TimeTraj()
trajectory, N = real_track.full_reference(N_horizon)
X0 = trajectory[:,0]

T = int(N*dt_ocp)
dt_sim = 0.02
Nsim = int(T / dt_sim)

lf =0.13  # distance from CoG to front wheels
lr =0.13 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ MODEL and OCP SOLVER ------------------

def Time_KinematiBicycleModel():
    model_name = "time_kin_bicycle"
    # STATES AND CONTROL
    # SX is scalar symbolic type
    x = SX.sym("x")
    y = SX.sym("y")
    psi = SX.sym("psi")
    x = vertcat(x, y, psi) # vertically concatenate
    v = SX.sym("v")
    delta = SX.sym("delta")
    u = vertcat(v, delta)
    beta = arctan(lr * tan(delta) / L) # slipping angle
    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    xdot = vertcat(x_dot, y_dot, psi_dot)
    # dynamics
    f_expl = vertcat(v * cos(psi+beta), v * sin(psi+beta),v *sin(beta) /lr) 
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name = model_name
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$\\psi$"]
    model.u_labels = [ "$v$", "$delta$"]
    return model

# ACADOS SOLVER SETTINGS
def CreateOcpSolver_TimeKin() -> AcadosOcp:
    ocp = AcadosOcp()
    model = Time_KinematiBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([1e2, 1e2, 1e2])
    R_mat = 2 * np.diag([1e0, 1e1])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,)) 
    ocp.model.cost_y_expr = vertcat( model.x[:2], model.x[2]+arctan(lr*tan(model.u[1])/L), model.u )
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([1e2, 1e2])*dt_ocp
    yref_e = np.array([4.0, 1.0])  
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:2]) 
  
    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([ -3, -np.deg2rad(60)])
    ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" 
    ocp.solver_options.tf = Tf
    return ocp

# ----------------- CLOSED LOOP SIMULATION ------------------

def ClosedLoopSimulationSIM():# LOOP OVER N SIM, UPDATING N OCP ACCORDINGLY
    ocp = CreateOcpSolver_TimeKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_nonlinear.json')

    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK' # FASTER THEN IRK 
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps =1 #int(dt_ocp/dt_sim)
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_nonlinear.json')
    
    #simulation
    N_horizon = acados_ocp_solver.N
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))                                                  
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((N_horizon+ 1, nx))
    predU = np.zeros((N_horizon+1, nu))
    yref_= np.zeros((N+N_horizon+1,nx))
    xcurrent = X0
    simX[0, :] = xcurrent
    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.1]))
    for stage in range(N_horizon+N):
        yref_[stage, :] = trajectory[:, stage] #np.concatenate((trajectory[:, stage], [0.0, 0.0]))
        
    for i in range(Nsim):
    # update reference
        k = int(dt_ocp/dt_sim)
        if  ((i%k)==0):
            for j in range(N_horizon): 
                
                acados_ocp_solver.set(j, "yref", np.concatenate((trajectory[:, i//k+j], [0.0, 0.0]))) #int(i/k)
            acados_ocp_solver.set(N_horizon, "yref", trajectory[:2, i//k+N_horizon]) #int(i/k)
        status = acados_ocp_solver.solve()
        if status != 0 and status != 2:
            print(f"ACADOS solver failed with status {status}")
        #x0 = acados_ocp_solver.get(0, "x")
        u0 = acados_ocp_solver.get(0, "u")
        
        xcurrent = acados_sim_solver.simulate(xcurrent, u0)
        simX[i+ 1, :] = xcurrent
        simU[i, :] = u0
        
        # update initial condition
        x0 = acados_ocp_solver.get(1, "x")
        acados_ocp_solver.set(0, "lbx", xcurrent) 
        acados_ocp_solver.set(0, "ubx", xcurrent) 
        print('x' , xcurrent)
        print('y', simU[i,:])

    plot_trajectory(simX, simU, yref_)

def ClosedLoopSimulationOCP(): # LOOP OVER N OCP, UPDATING N SIM ACCORDINGLY
    ocp = CreateOcpSolver_TimeKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_kin.json')

    sim = AcadosSim()
    sim.model = ocp.model    # use same model as OCP (or a different “plant” model)
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps =int(dt_ocp/dt_sim)
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_kin.json')

    #simulation
    N_horizon = acados_ocp_solver.N
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim+1 , nx))                                                  
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((N_horizon+ 1, nx))
    predU = np.zeros((N_horizon+1, nu))
    yref_= np.zeros((N+N_horizon+1,nx))

    xcurrent = X0
    simX[0, :] = xcurrent

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0]))
    for stage in range(N_horizon+N):
        yref_[stage, :] = trajectory[:, stage] #np.concatenate((trajectory[:, stage], [0.0, 0.0]))
        
    for i in range(N):
    # update reference
        for j in range(N_horizon): 
            acados_ocp_solver.set(j, "yref", np.concatenate((trajectory[:, i+j], [0.0, 0.0]))) #int(i/k)
        acados_ocp_solver.set(N_horizon, "yref", trajectory[:2, i+N_horizon]) 
        status = acados_ocp_solver.solve()
        if status != 0 and status != 2:
            print(f"ACADOS solver failed with status {status}")
        u0 = acados_ocp_solver.get(0, "u")
        
        for k in range (int(dt_ocp/dt_sim)):
            j = i*int(dt_ocp/dt_sim)
            xcurrent = acados_sim_solver.simulate(xcurrent, u0)
            simX[j+k+ 1, :] = xcurrent
            simU[j+k, :] = u0
        plot_trajectory(simX, simU, yref_)
        
        # update initial condition
        x0 = acados_ocp_solver.get(1, "x")
        acados_ocp_solver.set(0, "lbx", xcurrent) 
        acados_ocp_solver.set(0, "ubx", xcurrent) 
        print('x' , xcurrent)
        print('u', simU[i,:])
    plot_trajectory(simX, simU, yref_)

# ----------------- PLOTTING ------------------
def plot_trajectory(simX, simU, yref_):
    timestampsx = np.linspace(0,(Nsim+1)*dt_sim,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*dt_sim,Nsim)
    timestampsy = np.linspace(0,(N+1)*dt_ocp,N+1)

    plt.figure()
    plt.plot(simX[:,0],simX[:,1], label='Simulation')
    plt.plot(yref_[:N+1,0], yref_[:N+1,1], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["x", "y", "heading+slipping angle", "v", "steering angle"]
    for i in range(3):
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsy, yref_[:N+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    theta =  simX[:-1,2] +  arctan(lr * tan(simU[:,1]) / L) # define it better
    ax[2].plot(timestampsu, theta) # simX[:,3]
    ax[2].plot(timestampsy, yref_[:N+1,2], '--', label='Reference')
    ax[2].set_ylabel(labels[3])
    for i in range(2):
        ax[i + 3].plot(timestampsu, simU[:, i])
        ax[i + 3].set_ylabel(labels[i + 3])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ClosedLoopSimulationOCP()
    

