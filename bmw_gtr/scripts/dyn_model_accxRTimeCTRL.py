from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant, fabs
from reference_generation_v import *
from ref_time2spatial import *
from vehicle_params import VehicleParams

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.01
dt_sim = 0.005
dt_control = 0.02
N_horizon = 50
Tf =N_horizon * ds_ocp

nodes_to_visit =[141,97,125,150]

traj_gen = TrajectoryGeneration(ds=ds_ocp, N_horizon=50, v_max=0.6, v_min=0.3, 
                                  use_curvature_velocity=False, smooth_velocity=True)

y_ref, s_ref, kappa_ref = traj_gen.generating_spatial_reference(nodes_to_visit)
y_ref = y_ref.T
print(y_ref.shape)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = s_ref.shape[0] - N_horizon
S = N*ds_ocp
Nsim = 2*int(S/dt_sim)
print(y_ref[0,:])
print(s_ref[0])
X0 = np.hstack((y_ref[0, :4], y_ref[0, 5], 0.0, y_ref[0, 4], 0.0))
print(X0)

params = VehicleParams()
params.BoschCar()  

# ------------------ MODELING ------------------
def Spatial_DynamicBicycleModel():
    model_name = "SpatialDynamicBicycle_model"

    ## CasADi Model
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
    beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta #+np.pi/2
    beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))
    
    Fx_d = 1/2*params.ro* params.Cz* params.Az* vx*fabs(vx)
    # Lateral tire forces - simplified Pacejka # forget about them, try 
    Fc_f = -params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
    Fc_r = - params.Dcr * sin( params.Ccr * arctan( params.Bcr * beta_r) )

    Fyf = Fc_f*cos(delta) #+np.pi/2) # this is the same as -sin(delta)
    Fyr = Fc_r
    
    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + a +  (-Fx_d - Fc_f*sin(delta))/params.m# we consider that this is completly longitudial acceleration
    dvy   = - vx*omega+ (Fyf + Fyr)/params.m
    dpsi = omega
    domega    = (params.lf*Fyf - params.lr*Fyr)/params.I_z


    #Spatial dynamics dx/ds = f(x,u)
    sdot = (vx * cos(e_psi) - vy *sin(e_psi))/(1 - kappa(s)* e_y)
    
    dx_ds    = dX / (sdot)
    dy_ds    = dY / (sdot)
    dvx_ds    = dvx/ (sdot)
    dvy_ds    = dvy/ (sdot)
    dpsi_ds  = (dpsi) / (sdot)
    domega_ds  = (domega) / (sdot)
    d_e_psi = (dpsi)/(sdot) - kappa(s)
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


def TimeDynModel_accx():
    # States
    x   = SX.sym('x')
    y   = SX.sym('y')
    vx  = SX.sym('vx')
    vy  = SX.sym('vy')
    psi = SX.sym('psi')
    omega  = SX.sym('r') # yaw rate - psi dot
    x = vertcat(x, y, vx, vy, psi, omega)

    # Controls: acceleration a, steering delta
    delta = SX.sym('delta')
    a = SX.sym('a')
    u = vertcat (a, delta)
    
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    vx_dot = SX.sym("vx_dot")
    vy_dot = SX.sym("vy_dot")
    psi_dot = SX.sym("psi_dot")
    omega_dot   = SX.sym('omega_dot')
    xdot = vertcat(x_dot, y_dot, vx_dot, vy_dot, psi_dot, omega_dot)

    beta = arctan(vy/(vx+1e-5)) # slipping angle
    # Slip angles  -  aprox of small angles
    beta_f = - arctan((vy + params.lf*omega)/(vx+1e-5)) + delta #+np.pi/2
    beta_r = arctan((vy - params.lr*omega)/(vx+1e-5))
    
    Fx_d = 1/2*params.ro* params.Cz* params.Az* vx*fabs(vx)
    # Lateral tire forces - simplified Pacejka # forget about them, try 
    Fc_f = -params.Dcf * sin( params.Ccf * arctan(params.Bcf * beta_f) )
    Fc_r = -params.Dcr * sin( params.Ccr * arctan( params.Bcr * beta_r) )

    Fyf = Fc_f*cos(delta) #+np.pi/2) # this is the same as -sin(delta)
    Fyr = Fc_r


    # DYNAMICS
    dX   = vx*cos(psi) - vy*sin(psi)
    dY  = vx*sin(psi) + vy*cos(psi)
    dvx   = vy*omega + a  +  (-Fx_d - Fc_f*sin(delta))/params.m# we consider that this is completly longitudial acceleration
    dvy   = - vx*omega+ (Fyf + Fyr)/params.m
    dpsi = omega
    domega    = (params.lf*Fyf - params.lr*Fyr)/params.I_z


    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega)
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name        = 'time_dyn_bicyc_a'
    model.x           = x
    model.xdot        = xdot
    model.u           = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model

# ------------------ACADOS SOLVER SETTINGS ------------------
def CreateOcpSolver_SpatialDyn() -> AcadosOcp:
    ocp = AcadosOcp()
    model = Spatial_DynamicBicycleModel()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = 3 + nu
    ny_e = 3

    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([5*1e1,1e2,1e1])  
    R_mat =  np.diag([1e-1,1e-1])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1],model.x[4], model.u) #
  
    #terminal costs
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    Q_mat = np.diag([1e1,5*1e1])
    ocp.cost.W_e = Q_mat*ds_ocp
    yref_e = np.array([0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[0]+ arctan(model.x[5]/ (model.x[4]+1e-5)), model.x[1]) 
 
    ocp.parameter_values  = np.array([s_ref[0]])
    # set constraints on the input                                                                                                             
    ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
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
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = Tf
    return ocp


def CreateOcpSolver_TimeDyn() -> AcadosSim:
    ocp = AcadosOcp()
    model = TimeDynModel_accx()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat =  np.diag([5*1e2, 5*1e2, 1e1, 5*1e0, 1e1,1e0,1e-1,1e0]) #np.diag([5*1e0, 5*1e0, 1e1, 1e1, 5*1e1,1e2])  

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS" 
    ocp.cost.W = scipy.linalg.block_diag(Q_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4]+arctan(model.x[3]/(model.x[2]+1e-5)), model.x[5] , model.u)
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e =2 * np.diag([5*1e1, 5*1e0, 1e1, 1e1, 5*1e1,1e1]) *ds_ocp
    yref_e = np.array([10, 0.0, 0.0, 0.0, np.pi/2, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = model.x #vertcat(model.x[:4]) 

    # set constraints                                                                                                               
    ocp.constraints.lbu = np.array([-5,-np.deg2rad(45)])
    ocp.constraints.ubu = np.array([5, np.deg2rad(45)])
    ocp.constraints.idxbu = np.array([0,1])
    ocp.constraints.x0 = X0
    #ocp.constraints.lbx = np.array([-10, -10])
    #ocp.constraints.ubx = np.array([10, 10])
    #ocp.constraints.idxbx = np.array([2,3])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    return ocp

# ------------------CLOSED LOOP ------------------
def ClosedLoopSimulation():
    # Setup solvers
    ocp = CreateOcpSolver_SpatialDyn()
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_spatial1.json')

    sim_solver = CreateOcpSolver_TimeDyn()
    sim = AcadosSim()
    sim.model = sim_solver.model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim  # [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_spatial1.json')

    # Simulation settings
      # 100 Hz control frequency
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim + N_horizon + 1, nx))
    predU = np.zeros((Nsim + N_horizon, nu))

    # Initialization
    xcurrent = X0[2:]  # time-domain state [x, y, vx, vy, ψ, ω]
    xcurr_ocp = X0.copy()    # spatial-domain state [eψ, ey, x, y, vx, vy, ψ, ω]
    simX[0, :] = xcurr_ocp
    predX[0, :] = xcurr_ocp
    s_sim = s_ref[0]
    S_sim = np.array(s_sim)
    k = 0
    control_timer = 0.0
    u0 = np.zeros(nu)

    # Initialize OCP guesses
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurr_ocp)
        idx = min(stage, len(s_ref) - 1)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[idx]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.array([0.0, 0.0]))

    simulation_running = True
    i = 0

    # ---------------- MAIN LOOP ----------------
    while simulation_running:
        # Check if it’s time to compute control
        if control_timer >= dt_control - 1e-9:
            # Compute current s, eψ, ey from time-domain state
            s_now, epsi_now, ey_now = time2spatial(
                xcurrent[0], xcurrent[1], xcurrent[4], s_ref, y_ref[:, 2:5]
            )
  

            # Find nearest reference index
            idx_closest = int(np.argmin(np.abs(s_ref - s_now)))

            # Update horizon references
            for j in range(N_horizon):
                idx_stage = min(idx_closest + j, len(s_ref) - 1)
                yref_stage = np.concatenate(
                    (y_ref[idx_stage, :2], [y_ref[idx_stage, 5]], [0.0, 0.0])
                )
                acados_ocp_solver.set(j, "yref", yref_stage)
                acados_ocp_solver.set(j, "p", np.array([s_ref[idx_stage]]))

            # Terminal
            idx_term = min(idx_closest + N_horizon, len(s_ref) - 1)
            acados_ocp_solver.set(N_horizon, "yref", y_ref[idx_term, :2])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[idx_term]]))

            # Set initial state constraints for OCP
            x0_ocp = np.hstack((epsi_now, ey_now, xcurrent))
            acados_ocp_solver.set(0, "lbx", x0_ocp)
            acados_ocp_solver.set(0, "ubx", x0_ocp)
            acados_ocp_solver.set(0, "x", x0_ocp)

            # Solve OCP
            status = acados_ocp_solver.solve()
            if status != 0:
                print(f"[WARNING] ACADOS solver failed with status {status}")

            # Retrieve predicted trajectories
            try:
                for j in range(N_horizon):
                    predX[i + j, :] = acados_ocp_solver.get(j, "x")
                    predU[i + j, :] = acados_ocp_solver.get(j, "u")
                predX[i + N_horizon, :] = acados_ocp_solver.get(N_horizon, "x")
            except Exception:
                pass

            # Get optimal control
            u0 = acados_ocp_solver.get(0, "u").copy()
            control_timer -= dt_control  # reset control timer
            # Print progress
            print(f"s_sim = {s_sim:.3f}, idx_ref = {idx_closest}, control = {u0}")


        # -------- SIMULATION STEP --------
        simU[i, :] = u0
        xnext = acados_sim_solver.simulate(xcurrent, u0, s_sim)
        xcurrent = xnext.copy()

        # Update spatial position and store logs
        s_sim, epsi, ey = time2spatial(
            xcurrent[0], xcurrent[1], xcurrent[4], s_ref, y_ref[:, 2:5]
        )
        simX[i + 1, :] = np.hstack((epsi, ey, xcurrent))
        simX[i + 1, 0] += np.arctan(simX[i + 1, 5] / (simX[i + 1, 4]))
        simX[i + 1, -2] += np.arctan(simX[i + 1, 5] / (simX[i + 1, 4]))
        S_sim = np.append(S_sim, s_sim)

        # Update ocp initial guess for next iteration
        xcurr_ocp = np.hstack((epsi, ey, xcurrent))
        acados_ocp_solver.set(0, "lbx", xcurr_ocp)
        acados_ocp_solver.set(0, "ubx", xcurr_ocp)

        
        # Step forward
        i += 1
        control_timer += dt_sim

        # Stop condition
        if s_sim >= (S - 2 * ds_ocp) or i >= Nsim:
            simulation_running = False

    # ---------------- PLOTTING ----------------
    t = np.linspace(0, (i + 1) * dt_sim, i + 1)
    y_ref_time = reference_to_time(y_ref, t, S_sim)
    plot_states(simX, simU, y_ref_time, i)

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


def plot_states(simX, simU, y_ref,Nf):
    # Time vectors
    timestampsx = np.linspace(0, (Nf+1)*dt_sim, Nf+1)
    timestampsu = np.linspace(0, Nf*dt_sim, Nf)
    timestampsy = np.linspace(0, (N+1)*ds_ocp, N+1)

    # --- Trajectory Plot ---
    plt.figure()
    plt.plot(simX[:Nf+1,2], simX[:Nf+1,3], label=r'Trajectory', linewidth=2)
    plt.plot(y_ref[:Nf+1,2], y_ref[:Nf+1,3], '--', alpha=0.9, label=r'Reference', color="C1", linewidth=1.5)
    plt.xlabel(r'$x [m]$')
    plt.ylabel(r'$y[m]$')
    #plt.axis('equal')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis('equal')

    # --- Subplot 1: Lateral Error ---

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(timestampsx, simX[:Nf+1,1], color="#808000", linewidth=2)
    axs[0].plot(timestampsx,  y_ref[:Nf+1, 1], '--', color="C1", linewidth=1.5)   
    axs[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
    axs[0].set_title("Lateral Tracking Error", fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 2: Heading Error ---
    axs[1].plot(timestampsx, simX[:Nf+1, 0],label=r'Simulation',  color="#808000", linewidth=2)
    axs[1].plot(timestampsx,  y_ref[:Nf+1, 0], '--', label=r'Reference', color="C1", linewidth=1.5)   
    axs[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
    axs[1].set_title("Heading Tracking Error", fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].set_xlabel(r'Time$[s]$')
    axs[1].legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    # --- 2. States ---
    state_indices = [2, 3, 4, 5, 6, 7]  # x, y, vx, vy, theta, omega
    state_labels = [r'$x[m]$', r'$y[m]$', r'$v_x[\frac{m}{s}]$', r'$v_y[\frac{m}{s}]$', r'$\theta[rad]$', r'$\omega[\frac{rad}{s}]$']
    fig2, ax2 = plt.subplots(len(state_indices), 1, sharex=True, figsize=(6, 12))
    for j, idx in enumerate(state_indices):
        ax2[j].plot(timestampsx, simX[:Nf+1, idx],label = r'Simulation', color="C0", linewidth=2)
        #ax2[j].plot(timestampsx, y_ref[:Nf+1, idx], '--',  label = r'Reference', color="#ffd166", linewidth=1.5)
        ax2[j].set_ylabel(state_labels[j])
        ax2[j].grid(True, linestyle="--", alpha=0.4)
    ax2[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax2[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax2[4].set_title("Direction Angle", fontsize=12, fontweight='bold')
    ax2[-1].set_xlabel(r'Time$[s]$')
    ax2[-1].legend(fontsize=11, loc='upper right')
    ax2[-1].grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    

    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    labels = [r'$a[\frac{m}{s^2}]$', r'$\delta[rad]$'] 
    
    ax2[0].plot(timestampsu, simU[:Nf, 0],color="C2", linewidth=2) #color="#A52A2A"
    ax2[0].set_ylabel(labels[0])
    ax2[0].set_title("Acceleration", fontsize=12, fontweight='bold')
    ax2[0].grid(True, linestyle="--", alpha=0.4)

    ax2[1].plot(timestampsu, simU[:Nf, 1],color="C3", linewidth=2) # color="#A52A2A"
    ax2[1].set_ylabel(labels[1])
    ax2[1].set_xlabel(r'Time$[s]$')
    ax2[1].set_title("Steering Angle", fontsize=12, fontweight='bold')
    ax2[1].grid(True, linestyle="--", alpha=0.6)

    fig2.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show(block=True)

    plt.show()
if __name__ == "__main__":
    ClosedLoopSimulation()
