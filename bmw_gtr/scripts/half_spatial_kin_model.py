from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant
#from reference_trajectory_spatial import  *
from reference_trajectory_generation1 import  *
from ref_time2spatial import *
'''
this model is quite tricky and not completely correct
our reference is computed in space while both of our controllers work in time 
so when representing reference in graphs we are representing space behaviour
while for simulation we are representing time behaiour, this can be quite confusing 
the choice of dt_sim and dt_ocp is crutial 

the proportion between dt_sim and ds decides our speed
'''
# ------------------ VEHICLE PARAMS ------------------
# some values, should be a script for itself
lf = 0.13 # distance from CoG to front wheels
lr = 0.13 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
ds = 0.02
dt_ocp = 0.02
dt_sim = 0.02
N_horizon =50
Tf =N_horizon *ds
track = TrajectoryGeneration()
nodes_to_visit = [390,307, 377]
#nodes_to_visit = [91,125,141,91, 125,141,91,125,141]
y_ref, s_ref, kappa_ref = track.generating_spatial_reference(ds, nodes_to_visit)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)
X0 = y_ref[0,:]

Ns = s_ref.shape[0]
S = s_ref[-1] - N_horizon*ds
N = int(S/dt_ocp * X0[-2])
Nsim = int(S/dt_sim * X0[-2])

print(f"S", S)
print(f"Ns" , N)
print(f"Nsim", Nsim)






# ------------------ MODELING ------------------
def HalfSpatial_KinematiBicycleModel():
    model_name = "HalfSpatialKinBicycleModel"
    ## CasADi Model
    s = SX.sym('s')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v= SX.sym("v")
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    x = vertcat(e_psi, e_y, x, y, psi, v, s)

    # Controls: steering angle delta, acceleration a
    a= SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    v_dot = SX.sym("v_dot")
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    s_dot = SX.sym("s_dot")
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,v_dot,s_dot)

    beta = arctan(lr * tan(delta) / L) # slipping angle
    vx = v* cos(psi+beta)
    vy = v* sin(psi+beta)
    dpsi = v *sin(beta) / lr
    
    #Spatial dynamics dx/ds = f(x,u)
    sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta)*sin(e_psi))/(1 - kappa(s)* e_y)
    dx_ds    = vx 
    dy_ds    = vy 
    dv_ds    = a 
    dpsi_ds  = (dpsi) 
    d_e_psi = (dpsi) - kappa(s)*sdot
    d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) 

    f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, dv_ds, sdot)
    f_impl = xdot - f_expl

    # algebraic variables
    z = vertcat([])
    # parameters
    p = vertcat([]) 

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u 
    model.p = p
    model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$v$", "$s$"]
    model.u_labels = [ "$a$", "$delta$"]
    model.name = model_name
    return model


def HalfSpatial_KinematiBicycleModel1():
    model_name = "HalfSpatialKinBicycleModel1"
    ## CasADi Model
    s = SX.sym('s')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v= SX.sym("v")
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    x = vertcat(e_psi, e_y, x, y, psi, s)

    # Controls: steering angle delta, acceleration a
    #a= SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(v, delta)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    v_dot = SX.sym("v_dot")
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    s_dot = SX.sym("s_dot")
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, psi_dot,s_dot)

    beta = arctan(lr * tan(delta) / L) # slipping angle
    vx = v* cos(psi+beta)
    vy = v* sin(psi+beta)
    dpsi = v *sin(beta) / lr
    
    #Spatial dynamics dx/ds = f(x,u)
    sdot = (v *cos(beta) * cos(e_psi) - v *sin(beta)*sin(e_psi))/(1 - kappa(s)* e_y)
    dx_ds    = vx 
    dy_ds    = vy 
    #dv_ds    = a 
    dpsi_ds  = (dpsi) 
    d_e_psi = (dpsi) - kappa(s)*sdot
    d_e_y = (v *cos(beta)  * sin(e_psi) + v *sin(beta) * cos(e_psi)) 

    f_expl = vertcat(d_e_psi, d_e_y, dx_ds, dy_ds, dpsi_ds, sdot)
    f_impl = xdot - f_expl

    # algebraic variables
    z = vertcat([])
    # parameters
    p = vertcat([]) 

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u 
    model.p = p
    model.x_labels = ["$e_psi$", "$e_y$","$x$", "$y$", "$\\psi$", "$s$"]
    model.u_labels = [  "$v$",  "$delta$"]
    model.name = model_name
    return model

# ------------------ACADOS SOLVER SETTINGS ------------------
def CreateOcpSolver_HalfSpatialKin() -> AcadosOcp:
    ocp = AcadosOcp()
    model = HalfSpatial_KinematiBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 3 + nu
    ny_e = 3

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e3,5*1e2,1e1]) #1e3,5*1e2,1e5 
    R_mat =  np.diag([1e0,1e-1]) #1e-1,1e-3

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.x[-1], model.u)
    
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = Q_mat*dt_ocp
    yref_e = np.array([0.0, 0.0, 20.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:2], model.x[-1]) 

    # set constraints  - this is for the input                                                                                                             
    ocp.constraints.lbu = np.array([-2.5, -np.deg2rad(60)])
    ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # constraints on the states
    ocp.constraints.lbx = np.array([ -np.deg2rad(40), -0.5])
    ocp.constraints.ubx = np.array([ np.deg2rad(40), 0.5])
    ocp.constraints.idxbx = np.array([0,1])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 70
    ocp.solver_options.tf = Tf
    return ocp


# ------------------CLOSED LOOP ------------------
def ClosedLoopSimulation():
    #AcadosOcpSovler
    ocp = CreateOcpSolver_HalfSpatialKin()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_spatialtiral.json')

    #AcadosIntegrator
    
    sim = AcadosSim()
    sim.model = ocp.model   
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_spatialtrial.json')

    #simulation
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim +N_horizon+ 1, nx))
    predU = np.zeros((Nsim+N_horizon, nu))
    #initialization
    xcurrent = X0
    print('x0', X0)
    predX[0,:] = xcurrent
    simX[0, :] = xcurrent


    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurrent)
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.0])) #warm start
    
    for i in range(Nsim):
        # Reference for horizon
        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[j+i,:2],[y_ref[j+i,-1]],[0.0, 0.0])))
        acados_ocp_solver.set(N_horizon, "yref", np.concatenate((y_ref[i+N_horizon,:2],[y_ref[i+N_horizon,-1]])))

        # SOLVE OCP PROBLEM
        status = acados_ocp_solver.solve()
        if status != 0:
            print(f"ACADOS solver failed with status {status}")
        
        for j in range(N_horizon):
            predX[i+j,:] = acados_ocp_solver.get(j, "x")
            predU[i+j,:] = acados_ocp_solver.get(j, "u")
        predX[i+N_horizon,:] = acados_ocp_solver.get(N_horizon, "x")
        
        # update initial condition - move to the next state
        u0 = acados_ocp_solver.get(0, "u")
        xcurrent = acados_sim_solver.simulate(xcurrent, u0) 
        simU[i, :] = u0
        simX[i + 1, :] = xcurrent
        simX[i + 1 , 0] = xcurrent[0] +arctan (lr*tan(simU[i, 1])/L) # representation of epsi 
        simX[i + 1, -3] = xcurrent[-3] +arctan (lr*tan(simU[i, 1])/L) # representation of theta 

        
        acados_ocp_solver.set(0, "lbx",xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        print('x' , xcurrent)
        print('y',y_ref[i+1,:])
        print('u', simU[i,:])
    plot_trajectory(simX, simU, y_ref)
    

def plot_trajectory(simX, simU, y_ref):
    timestampsx = np.linspace(0,(Nsim+1)*dt_sim,Nsim+1)
    timestampsu = np.linspace(0,(Nsim)*dt_sim,Nsim)
    timestampsy = np.linspace(0,(Ns)*ds,Ns)

    plt.figure()
    plt.plot(simX[:,2],simX[:,3], label='Simulation')
    plt.plot(y_ref[:,2], y_ref[:,3], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y", "x", "y", "theta", "v", "s" ,"a" ,"delta"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:, i])
        ax[i].plot(timestampsy, y_ref[:,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].set_xlabel("time[s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ClosedLoopSimulation()
