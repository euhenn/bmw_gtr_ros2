from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import patches
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant

from ref_time2spatial import *
from reference_trajectory_generation1 import  *

# some values, should be a script for itself
lf = 0.13 # distance from CoG to front wheels
lr = 0.13 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.1
dt_sim = 0.05
N_horizon = 50
Tf =N_horizon *ds_ocp
ds = 0.025
track = TrajectoryGeneration()
nodes_to_visit = [230,307, 377,418]
#nodes_to_visit = [91,125,141,91,125,141]
y_ref, s_ref, kappa_ref = track.generating_spatial_reference(ds, nodes_to_visit)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = s_ref.shape[0] - N_horizon

S = N*ds
Nsim = 2*int(S/dt_sim)
print(y_ref[0,:])
print(s_ref[0])
X0 = y_ref[0, :]

# ------------------ MODELING ------------------
def spatial_bicycle_model_ocp():
    model_name = "SpatialBicycleOCP"

    ## CasADi Model
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
    model.name = model_name
    return model

def time_bicycle_model_sim():
    model_name = "TimeBicycleSim"

    # SX is scalar symbolic type
    x = SX.sym("x")
    y = SX.sym("y")
    v = SX.sym("v")
    psi = SX.sym("psi")
    x = vertcat(x, y, psi, v) # vertically concatenate
    a = SX.sym("a")
    delta = SX.sym("delta")
    u = vertcat(a, delta)
    beta = arctan(lr * tan(delta) / L) # slipping angle
    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    v_dot = SX.sym("v_dot")
    psi_dot = SX.sym("psi_dot")
    xdot = vertcat(x_dot, y_dot, psi_dot, v_dot)
    # dynamics
    f_expl = vertcat(v * cos(psi+beta), v * sin(psi+beta),v *sin(beta) /lr, a) 
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name = model_name
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$\\psi$", "$v$"]
    model.u_labels = [ "$a$", "$delta$"]
    return model

# ------------------ACADOS SOLVER SETTINGS ------------------
def create_ocp_solver() -> AcadosOcp:
    ocp = AcadosOcp()
    model = spatial_bicycle_model_ocp()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 2 + nu
    ny_e = 2

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e3,5*1e2])  
    R_mat =  np.diag([1e0,1e-3])

    ocp.parameter_values  = np.array([s_ref[0]])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.u) #
  
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = Q_mat*ds_ocp
    yref_e = np.array([0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:2]) 

    # set constraints  - this is for the input                                                                                                             
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

def create_sim_solver() -> AcadosOcp:
    ocp = AcadosOcp()
    model = time_bicycle_model_sim()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = 2 * np.diag([1e1,1e1, 1e1, 5*1e2])  #ellipse
    #Q_mat = 2 * np.diag([1e2, 1e2 , 1e3, 1e2])  #s_surve

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((nx,)) #ny
    ocp.model.cost_y_expr = vertcat( model.x[:2], model.x[2]+arctan(lr*tan(model.u[1])/L) , model.x[3])
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([1e2, 1e2 , 1e1])*dt_sim
    yref_e = np.array([10.0, 0.0, 0.0]) #ellipse path
    #yref_e = np.array([50.0, 0.0, 0.0])  #s curve path
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:3]) 
  
    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([ -3, -np.deg2rad(60)])
    ocp.constraints.ubu = np.array([3, np.deg2rad(60)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" # "PARTIAL_CONDENSING_HPIPM" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP" #SQP_RTI
    ocp.solver_options.tf = Tf
    return ocp



# ------------------INTEGATION BY HAND ERK4------------------
def one_step_interation(ey, epsi, psi, v, a, beta):
    sdot = (v *cos(beta) * cos(epsi) - v *sin(beta) *sin(epsi))/(1 - kappa(scurrent)* ey)
    dx =  v* cos(psi+beta)/ sdot
    dy =  v * sin(psi+beta)/ sdot
    dv =  a/ sdot
    dpsi = v *sin(beta) / lr / sdot
    depsi = dpsi - kappa(scurrent) 
    dey = (v *cos(beta) * sin(epsi) + v *sin(beta) * cos(epsi)) / sdot
    return depsi,dey,dx,dy,dpsi,dv 

def Integrator(xcurrent, u, scurrent):
    print("Integration by ERK4")
    h = 0.5

    a = u[0]
    beta = arctan(lr * tan(u[1]) / L) 
    epsi = xcurrent[0]
    ey = xcurrent[1]
    x = xcurrent[2]
    y = xcurrent[3]
    psi = xcurrent[4]
    v  = xcurrent[5]
    
    depsi1,dey1,dx1,dy1,dpsi1,dv1 = one_step_interation(ey, epsi, psi, v, a, beta)
    depsi2,dey2,dx2,dy2,dpsi2,dv2 = one_step_interation(ey+ h *ds_sim *dey1,epsi+ h *ds_sim *depsi1 , psi + h *ds_sim *dpsi1, v + h *ds_sim *dv1, a, beta)
    depsi3,dey3,dx3,dy3,dpsi3,dv3 = one_step_interation(ey+ h *ds_sim *dey2, epsi+ h *ds_sim *depsi2, psi + h *ds_sim *dpsi2, v + h *ds_sim *dv2, a, beta)
    depsi4,dey4,dx4,dy4,dpsi4,dv4 = one_step_interation(ey+ h *ds_sim *dey3, epsi+ ds_sim *depsi3, psi + ds_sim *dpsi3, v + ds_sim *dv3, a, beta)
    #next
    epsi_next = epsi + ds_sim * (depsi1+2*depsi2+2*depsi3+depsi4)/6
    ey_next = ey + ds_sim * (dey1+2*dey2+2*dey3+dey4)/6
    x_next = x + ds_sim * (dx1+2*dx2+2*dx3+dx4)/6
    y_next = y + ds_sim * (dy1+2*dy2+2*dy3+dy4)/6
    v_next = v + ds_sim * (dv1+2*dv2+2*dv3+dv4)/ 6
    psi_next = psi + ds_sim * (dpsi1+2*dpsi2+2*dpsi3+dpsi4)/ 6
    x_current = np.stack((epsi_next,ey_next, x_next,y_next,v_next,psi_next))

    return x_current

# ------------------CLOSED LOOP ------------------
def closed_loop_simulation():
    #AcadosOcpSovler
    ocp = create_ocp_solver()
    model = ocp.model
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_rt2.json')

    #AcadosIntegrator
    sim_solver= create_sim_solver()
    sim = AcadosSim()
    sim.model = sim_solver.model   
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim # simulation step size [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_rt2.json') # instead of it we can use EKR that we made

    #simulation
    simulaton_running = True
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim +N_horizon+ 1, nx))
    predU = np.zeros((Nsim+N_horizon, nu))

    #initialization 
    xcurrent = y_ref[0, 2:]
    print(xcurrent)
    xcurr_ocp = X0
    predX[0,:] = xcurr_ocp
    simX[0, :] = xcurr_ocp
    s_sim = s_ref[0]
    S_sim = np.array(s_sim)
    s_prev = - ds_ocp
    k = 0

    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurr_ocp)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[stage]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u",  np.array([0.0, 0.1])) #warm start

    simulation_running = True
    i = 0
    while simulation_running: 
        # Reference for horizon
        # compute ssim - what is the arc of length in simulation
        #s_sim = s_sim + dt_sim * xcurrent[-1]
        if (s_sim- s_prev >= ds_ocp) or (s_sim > s_ref[k+1]):
            for j in range(N_horizon):
                acados_ocp_solver.set(j, "yref", np.concatenate((y_ref[j+k,:2],[0.0, 0.0])))
                acados_ocp_solver.set(j, "p", np.array([s_ref[k + j]]))
            acados_ocp_solver.set(N_horizon, "yref", y_ref[k+N_horizon,:2])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[k+ N_horizon]]))
            s_prev = s_sim
            k = k+1
            # SOLVE OCP PROBLEM
            status = acados_ocp_solver.solve()
            if status != 0:
                print(f"ACADOS solver failed with status {status}")
            
            for j in range(N_horizon):
                predX[i+j,:] = acados_ocp_solver.get(j, "x")
                predU[i+j,:] = acados_ocp_solver.get(j, "u")
            predX[i+N_horizon,:] = acados_ocp_solver.get(N_horizon, "x")
            # get the control
            u0 = acados_ocp_solver.get(0, "u")
        simU[i, :] = u0
        xcurrent = acados_sim_solver.simulate(xcurrent, u0) 
        
        s_sim,epsi ,ey= time2spatial(xcurrent[0], xcurrent[1], xcurrent[2],s_ref,y_ref[:,2:5])
        simX[i + 1, :] = np.hstack((epsi,ey,xcurrent))
        simX[i + 1 , 0] =  simX[i + 1, 0]+arctan (lr*tan(simU[i, 1])/L) # representation of epsi + Beta 
        simX[i + 1, -2] =simX[i + 1, -2] + arctan (lr*tan(simU[i, 1])/L) # representation of theta 
        S_sim = np.append(S_sim, s_sim)
        xcurr_ocp = np.hstack((epsi,ey,xcurrent))
        acados_ocp_solver.set(0, "lbx",xcurr_ocp)
        acados_ocp_solver.set(0, "ubx", xcurr_ocp)
                
        print('SREF' , s_ref[k])
        print('Ssim' , s_sim)
        #print('S' ,s_[-1])

        i = i+1
        if (s_sim>= (S- (N_horizon+2)*ds_ocp)):
            simulation_running= False 
    t = np.linspace(0,(i+1)*dt_sim,i+1)

    y_ref_time  = reference_to_time(y_ref, t, S_sim)
    plot_states(simX, simU, y_ref_time, i)
   

def plot_states(simX, simU, y_ref_time,Nf):
    timestampsx = np.linspace(0,(Nf+1)*dt_sim,Nf+1)
    timestampsu = np.linspace(0,(Nf)*dt_sim,Nf)
    timestampsy = np.linspace(0,(N+1)*ds_ocp,N+1)
    Ny = timestampsy.shape[0]

    plt.figure()
    plt.plot(simX[:Nf+1,2],simX[:Nf+1,3], label='Simulation')
    plt.plot(y_ref[:Ny+1,2], y_ref[:Ny+1,3], '--', alpha=0.9 , c = "orange" ,label='Reference')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.legend()

    nx = simX.shape[1]
    nu = simU.shape[1]
    fig, ax = plt.subplots(nx+nu, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["e_psi", "e_y", "x", "y", "theta", "v", "a", "delta"]
    for i in range(nx):
        ax[i].plot(timestampsx, simX[:Nf+1, i])
        ax[i].plot(timestampsx, y_ref[:Nf+1,i], '--', label='Reference')
        ax[i].set_ylabel(labels[i])
    for i in range(nu):
        ax[i + nx].plot(timestampsu, simU[:Nf, i])
        ax[i + nx].set_ylabel(labels[i + nx])
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show(block=True)
    

def reference_to_time(y_ref, time, s_sim):
    y_ref_time = np.zeros((len(time), y_ref.shape[1]))
    for j in range(y_ref.shape[1]):
        y_ref_time[:, j] = np.interp(
            s_sim,      # arc length values at each time step
            s_ref,              # arc length coordinates of reference points
            y_ref[:, j],        # reference values at those arc lengths
            left=y_ref[0, j],   # value for points before first reference point
            right=y_ref[-1, j]  # value for points after last reference point
        )
    return y_ref_time


if __name__ == "__main__":
    closed_loop_simulation()
