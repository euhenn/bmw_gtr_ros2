from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import patches
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant

from ref_time2spatial import *
from reference_generation_v import  *

# some values, should be a script for itself
lf = 0.1335 # distance from CoG to front wheels
lr = 0.1335 #distance form CoG to rear wheels
L = lr + lf # wheel base

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.01
dt_sim = 0.005
N_horizon = 50
Tf =N_horizon *ds_ocp
ds = 0.01
dt_control = 0.02
nodes_to_visit = [230,307]
nodes_to_visit = [141,91]#,125,150] #141,91,
nodes_to_visit = [397,307,377]
nodes_to_visit = [330,307]

traj_gen = TrajectoryGeneration(ds=ds, N_horizon=50, v_max=0.6, v_min=0.3, 
                                  use_curvature_velocity=False, smooth_velocity=True)

y_ref, s_ref, kappa_ref = traj_gen.generating_spatial_reference(nodes_to_visit)


y_ref = y_ref.T
print(y_ref.shape)
kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = s_ref.shape[0] - N_horizon

S = N*ds
Nsim = 2*int(S/dt_sim)
print(y_ref[0,:])
print(s_ref[0])
X0 = y_ref[0,:]

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
    Q_mat = np.diag([1e2,5*1e1])  
    R_mat =  np.diag([1e0,1e-2])

    ocp.parameter_values  = np.array([s_ref[0]])

    #path const
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[0]+arctan(lr*tan(model.u[1])/L),model.x[1], model.u) #
  
    #terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.diag([1e3,5*1e2])*ds_ocp
    yref_e = np.array([0.0, 0.0]) 
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = vertcat(model.x[:2]) 

    # set constraints  - this is for the input                                                                                                             
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

# ==========================================================
#                     CSV EXPORT CLASS
# ==========================================================

class DataLogger:
    def save_csv(self, simX, simU, S_sim, y_ref_s, filename="results.csv"):
        data = np.hstack([S_sim.reshape(-1, 1), simX, simU])
        header = "s," + ",".join(
            [f"x{i}" for i in range(simX.shape[1])] +
            [f"u{i}" for i in range(simU.shape[1])]
        )
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        print(f"[CSV] Saved {filename}")


# ------------------CLOSED LOOP (modified) ------------------
def closed_loop_simulation():
    # --- Create solvers ---
    ocp = create_ocp_solver()
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_rt2.json')

    sim_solver = create_sim_solver()
    sim = AcadosSim()
    sim.model = sim_solver.model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_rt2.json')

    # --- Allocate memory ---
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim + N_horizon + 1, nx))
    predU = np.zeros((Nsim + N_horizon, nu))

    # --- Initialization ---
    xcurrent = y_ref[0, 2:].copy()
    xcurr_ocp = X0.copy()
    predX[0, :] = xcurr_ocp
    simX[0, :] = xcurr_ocp
    s_sim = s_ref[0]
    S_sim = np.array(s_sim)

    # Initial warm start
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", xcurr_ocp)
        idx = min(stage, len(s_ref) - 1)
        acados_ocp_solver.set(stage, "p", np.array([s_ref[idx]]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.array([0.0, 0.1]))

    # --- Performance tracking ---
    solve_times = []
    loop_times = []
    qp_iterations = []
    import time
    last_time = time.time()

    # --- Control scheduling ---
    control_timer = 0.0
    u0 = np.zeros(nu)
    i = 0
    simulation_running = True
    saver = DataLogger()

    # -------------------- MAIN LOOP --------------------
    while simulation_running:

        loop_start = time.time()   # timing start

        # ================= MPC UPDATE ==================
        if control_timer >= dt_control - 1e-12:

            t0_solve = time.time()    # start MPC solve timing

            # Compute spatial error from time state
            s_now, epsi_now, ey_now = time2spatial(
                xcurrent[0], xcurrent[1], xcurrent[2],
                s_ref, y_ref[:, 2:5]
            )

            # Find nearest index in the path
            idx_closest = int(np.argmin(np.abs(s_ref - s_now)))

            # Set horizon reference
            for j in range(N_horizon):
                idx_stage = min(idx_closest + j, len(s_ref) - 1)
                acados_ocp_solver.set(j, "yref",
                    np.concatenate((y_ref[idx_stage, :2], [0.0, 0.0])))
                acados_ocp_solver.set(j, "p", np.array([s_ref[idx_stage]]))
            idx_term = min(idx_closest + N_horizon, len(s_ref) - 1)
            acados_ocp_solver.set(N_horizon, "yref", y_ref[idx_term, :2])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[idx_term]]))

            # Lock initial state (spatial)
            x0_ocp = np.hstack((epsi_now, ey_now, xcurrent))
            acados_ocp_solver.set(0, "lbx", x0_ocp)
            acados_ocp_solver.set(0, "ubx", x0_ocp)
            acados_ocp_solver.set(0, "x",  x0_ocp)

            # --------- Solve OCP ----------
            status = acados_ocp_solver.solve()
            solve_time = time.time() - t0_solve

            solve_times.append(solve_time)


            # --- Extract QP iterations (robust for all solvers) ---
            qp_iter_raw = acados_ocp_solver.get_stats("qp_iter")
            sqp_iter_raw = acados_ocp_solver.get_stats("sqp_iter")

            # Prefer QP iteration count if available
            if isinstance(qp_iter_raw, (list, tuple, np.ndarray)) and len(qp_iter_raw) > 0:
                qp_iterations.append(int(qp_iter_raw[-1]))

            # Otherwise fall back to SQP iterations
            elif isinstance(sqp_iter_raw, (list, tuple, np.ndarray)):
                qp_iterations.append(int(sqp_iter_raw[-1]))

            else:
                qp_iterations.append(0)



            if status != 0:
                print(f"[WARNING] acados solver returned status {status}")

            # Extract and store predicted trajectory
            for j in range(N_horizon):
                predX[i + j, :] = acados_ocp_solver.get(j, "x")
                predU[i + j, :] = acados_ocp_solver.get(j, "u")
            predX[i + N_horizon, :] = acados_ocp_solver.get(N_horizon, "x")

            # Control output
            u0 = acados_ocp_solver.get(0, "u").copy()

            # Reset timer
            control_timer -= dt_control

        # ================= SIMULATION STEP =================
        simU[i, :] = u0
        x_next = acados_sim_solver.simulate(xcurrent, u0)
        xcurrent = x_next.copy()

        # Convert to spatial state
        s_sim, epsi_sim, ey_sim = time2spatial(
            xcurrent[0], xcurrent[1], xcurrent[2],
            s_ref, y_ref[:, 2:5]
        )
        simX[i + 1, :] = np.hstack((epsi_sim, ey_sim, xcurrent))
        simX[i + 1, 0] += arctan(lr * tan(simU[i, 1]) / L)
        simX[i + 1, -2] += arctan(lr * tan(simU[i, 1]) / L)

        S_sim = np.append(S_sim, s_sim)

        # Initial guess update for next rt-MPC step
        xcurr_ocp = np.hstack((epsi_sim, ey_sim, xcurrent))
        acados_ocp_solver.set(0, "lbx", xcurr_ocp)
        acados_ocp_solver.set(0, "ubx", xcurr_ocp)

        # Time bookkeeping
        i += 1
        control_timer += dt_sim

        loop_times.append(time.time() - loop_start)

        if s_sim >= (S - 2 * ds_ocp) or i >= Nsim:
            simulation_running = False

    # ---------------- PERFORMANCE SUMMARY ----------------
    
    
    print("\n========== MPC PERFORMANCE SUMMARY ==========")
    print(f"Total MPC calls: {len(solve_times)}")
    print(f"Average MPC solve time: {np.mean(solve_times)*1000:.3f} ms")
    print(f"Max MPC solve time:     {np.max(solve_times)*1000:.3f} ms")
    print(f"Min MPC solve time:     {np.min(solve_times)*1000:.3f} ms")
    print(f"Average QP iterations:  {np.mean(qp_iterations):.2f}")
    print(f"Average loop time:      {np.mean(loop_times)*1000:.3f} ms")
    print(f"Total simulation time:  {np.sum(loop_times):.3f} s")
    print(f"Real-time ratio (=loop_time/dt_control): "
          f"{np.mean(loop_times)/dt_control:.3f}")
    print("=============================================\n")
    

    # ---------------- PLOTTING ----------------
    #t = np.linspace(0, (i+1) * dt_sim, i+1)
    y_ref_s = reference_at_s(y_ref, S_sim,s_ref)
    saver.save_csv(simX[:i+1,:], simU[:i+1,:], S_sim, y_ref_s)
    plot_states(simX, simU, y_ref_s, i)



def plot_states(simX, simU, y_ref_s,Nf):
    timestampsx = np.linspace(0,(Nf+1)*dt_sim,Nf+1)
    timestampsu = np.linspace(0,(Nf)*dt_sim,Nf)
    timestampsy = np.linspace(0,(N+1)*ds_ocp,N+1)
    Ny = timestampsy.shape[0]

    plt.figure(figsize=(8, 6))
    plt.plot(simX[:Nf+1,2],simX[:Nf+1,3], label=r'Simulation', linewidth = 2)
    plt.plot(y_ref_s[:,2], y_ref_s[:,3], '--', alpha=0.9 , c = "orange" ,label=r'Reference', linewidth = 1.5)
    plt.scatter(simX[0,2], simX[0,3], color='green', s=80, marker='o', label='Start')
    plt.scatter(simX[Nf,2], simX[Nf,3], color='red', s=80, marker='x', label='End')
    plt.xlabel(r'$x[m]$')
    plt.ylabel(r'$y[m]$')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axis('equal')

    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Subplot 1: Lateral Error ---
    axs[0].plot(timestampsx, simX[:Nf+1,1], color="#808000", linewidth=2)
    axs[0].plot(timestampsx,  y_ref_s[:Nf+1, 1], '--', color="C1", linewidth=1.5)   
    axs[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
    axs[0].set_title("Lateral Tracking Error", fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Subplot 2: Heading Error ---
    axs[1].plot(timestampsx, simX[:Nf+1, 0],label=r'Simulation',  color="#808000", linewidth=2)
    axs[1].plot(timestampsx,  y_ref_s[:Nf+1, 0], '--', label=r'Reference', color="C1", linewidth=1.5)   
    axs[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
    axs[1].set_title("Heading Tracking Error", fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].set_xlabel(r'Arc length $[m]$')
    axs[1].legend(fontsize=11, loc='lower right')
    plt.tight_layout()


    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8,14))
    labels = [r'$x[m]$', r'$y[m]$', r'$v[\frac{m}{s}]$', r'$\theta[rad]$']
    for j in range(2):
        ax[j].plot(timestampsx, simX[:Nf+1, j+2], linewidth = 2)
        ax[j].plot(timestampsx,  y_ref_s[:Nf+1, j+2], '--', label='Reference', color = "C1", linewidth = 1.5)
        ax[j].set_ylabel(labels[j])
        ax[j].grid(True, linestyle="--", alpha=0.4)
    ax[2].plot(timestampsx, simX[:Nf+1, 5], linewidth=2) #, color="#3498DB"
    ax[2].plot(timestampsx, y_ref_s[:Nf+1,5], '--', color="C1", linewidth=1.5)
    ax[2].set_ylabel(labels[2])
    ax[2].grid(True, linestyle="--", alpha=0.4)
    theta =  simX[:Nf+1,4] +  arctan(lr * tan(simU[:Nf+1,1]) / L) # define it better
    ax[3].plot(timestampsx, simX[:Nf+1,4], label = r'Simulation' , color="#3498DB", linewidth=2) # simX[:,3]
    ax[3].plot(timestampsx, y_ref_s[:Nf+1,4], '--', label = r'Reference' ,  color="C1", linewidth=1.5)
    ax[3].set_ylabel(labels[3])
    ax[3].set_xlabel(r'Arc length $[m]$')
    ax[3].legend(fontsize=11, loc='lower right')
    ax[3].grid(True, linestyle="--", alpha=0.4)

    ax[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax[3].set_title("Direction Angle", fontsize=12, fontweight='bold')
    
    
    
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    labels = [r'$a[\frac{m}{s^2}]$', r'$\delta[rad]$'] 
    
    ax2[0].plot(timestampsu, simU[:Nf, 0],color="C2", linewidth=2) #color="#A52A2A"
    ax2[0].set_ylabel(labels[0])
    ax2[0].set_title("Acceleration", fontsize=12, fontweight='bold')
    ax2[0].grid(True, linestyle="--", alpha=0.4)

    ax2[1].plot(timestampsu, simU[:Nf, 1],color="C3", linewidth=2) # color="#A52A2A"
    ax2[1].set_ylabel(labels[1])
    ax2[1].set_xlabel(r'Arc length $[m]$')
    ax2[1].set_title("Steering Angle", fontsize=12, fontweight='bold')
    ax2[1].grid(True, linestyle="--", alpha=0.6)

    fig2.tight_layout(rect=[0, 0.05, 1, 1])
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

def reference_at_s(y_ref,s_sim, s_ref):
    Y = np.zeros((len(s_sim), y_ref.shape[1]))
    for j in range(y_ref.shape[1]):
        Y[:, j] = np.interp(s_sim, s_ref, y_ref[:, j])
    return Y



if __name__ == "__main__":
    closed_loop_simulation()


