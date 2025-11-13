from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant, fabs, tanh   # <-- added tanh
from reference_trajectory_dynamic import *
from ref_time2spatial import *
from vehicle_params import VehicleParams

# ------------------ INITIAL CONFIGURATION ------------------
ds_ocp = 0.01
dt_sim = 0.01
dt_control = 0.02
N_horizon = 50
Tf = N_horizon * ds_ocp

straight_line = StraightLineTrajectory()
ellipse = EllipseTrajectory()
scurve = SCurveTrajectory()

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


X0 = np.hstack((y_ref[0, :],  0.0))   # <-- ensure length = 9
print(X0)
a_max = 1

params = VehicleParams()
params.BoschCar()

# ------------------ MODELING ------------------
def Spatial_DynamicBicycleModel_withDeltaFz():
    """
    Spatial-domain dynamic double-bicycle model with delta_Fz (normal load transfer)
    State order: e_psi, e_y, x, y, vx, vy, psi, omega, delta_Fz
    Controls: a, delta
    Parameter: s (arc length) used for kappa(s) interpolant
    """
    model_name = "SpatialDynamicBicycle_deltaFz"

    # symbolic states
    s = SX.sym('s')
    e_psi = SX.sym('e_psi')
    e_y = SX.sym('e_y')
    x_pos = SX.sym('x')
    y_pos = SX.sym('y')
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    psi = SX.sym('psi')
    omega = SX.sym('r')
    delta_Fz = SX.sym('delta_Fz')

    x = vertcat(e_psi, e_y, x_pos, y_pos, vx, vy, psi, omega, delta_Fz)

    # controls
    a = SX.sym('a')
    delta = SX.sym('delta')
    u = vertcat(a, delta)

    # xdot (time derivatives) as symbolic placeholders - casadi requires xdot vector for implicit model
    e_psi_dot = SX.sym('e_psi_dot')
    e_y_dot = SX.sym('e_y_dot')
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    vx_dot = SX.sym('vx_dot')
    vy_dot = SX.sym('vy_dot')
    psi_dot = SX.sym('psi_dot')
    omega_dot = SX.sym('omega_dot')
    delta_Fz_dot = SX.sym('delta_Fz_dot')
    xdot = vertcat(e_psi_dot, e_y_dot, x_dot, y_dot, vx_dot, vy_dot, psi_dot, omega_dot, delta_Fz_dot)

    # short names to params
    lf = params.lf
    lr = params.lr

    # vehicle kinematics (wheelframe velocities)
    vy_f = vy + lf * omega
    vy_r = vy - lr * omega
    vx_f = vx
    vx_r = vx

    # velocities in wheel frames (front accounts for steering)
    vc_f = vy_f * cos(delta) - vx_f * sin(delta)   # lateral component front in wheel frame
    vc_r = vy_r
    vl_f = vy_f * sin(delta) + vx_f * cos(delta)   # longitudinal component front in wheel frame
    vl_r = vx_r

    # slip angles
    beta_f = arctan(vc_f / (vl_f + 1e-5))
    beta_r = arctan(vc_r / (vl_r + 1e-5))

    # aero drag
    Fx_d = 0.5 * params.ro * params.Cz * params.Az * vx * fabs(vx)

    # lateral tire forces (simplified Pacejka)
    Fc_f = - params.mi * params.Dcf * sin(params.Ccf * arctan((params.Bcf * beta_f) / params.mi))
    Fc_r = - params.mi * params.Dcr * sin(params.Ccr * arctan((params.Bcr * beta_r) / params.mi))

    # static normal loads (nominal)
    Fzf_0 = params.m * params.g * lr / (params.L)
    Fzr_0 = params.m * params.g * lf / (params.L)

    # include delta_Fz effect on normal loads
    Fzf = Fzf_0 - delta_Fz
    Fzr = Fzr_0 + delta_Fz

    # longitudinal traction/braking: saturate via tanh(a/a_max)
    gamma = tanh(a / a_max)   # <-- tanh imported
    Fl_f = gamma * params.mi * Fzf
    Fl_r = gamma * params.mi * Fzr

    # wheel forces (combine longitudinal and lateral, account for steering at front)
    Fx_f = Fl_f * cos(delta) - Fc_f * sin(delta)
    Fx_r = Fl_r
    Fx = Fx_f + Fx_r

    Fy_f = Fl_f * sin(delta) + Fc_f * cos(delta)
    Fy_r = Fc_r

    # pitch/roll induced vertical transfer (same algebraic formula as your time model)
    deltaz = - params.h * Fx / (2 * params.L)
    tau_z = 0.5
    ddelta_Fz = (deltaz - delta_Fz) / tau_z

    # time-domain dynamics (as in your time model)
    dX = vx * cos(psi) - vy * sin(psi)
    dY = vx * sin(psi) + vy * cos(psi)
    dvx = vy * omega + (Fx - Fx_d) / params.m
    dvy = - vx * omega + (Fy_f + Fy_r) / params.m
    dpsi = omega
    domega = (lf * Fy_f - lr * Fy_r) / params.I_z

    # spatial rate sdot (arc-length derivative)
    sdot = (vx * cos(e_psi) - vy * sin(e_psi)) / (1 - kappa(s) * e_y + 1e-8)

    # convert time derivatives to spatial derivatives dx/ds = (dx/dt) / sdot
    dx_ds = dX / (sdot)
    dy_ds = dY / (sdot)
    dvx_ds = dvx / (sdot)
    dvy_ds = dvy / (sdot)
    dpsi_ds = dpsi / (sdot)
    domega_ds = domega / (sdot)
    d_e_psi_ds = dpsi / (sdot) - kappa(s)
    d_e_y_ds = (vx * sin(e_psi) + vy * cos(e_psi)) / (sdot)
    d_deltaFz_ds = ddelta_Fz / (sdot)

    f_expl = vertcat(
        d_e_psi_ds,
        d_e_y_ds,
        dx_ds,
        dy_ds,
        dvx_ds,
        dvy_ds,
        dpsi_ds,
        domega_ds,
        d_deltaFz_ds
    )
    f_impl = xdot - f_expl

    # Build AcadosModel
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = vertcat(s)
    model.name = model_name
    model.x_labels = ["$e_\\psi$", "$e_y$", "$x$", "$y$", "$v_x$", "$v_y$", "$\\psi$", "$r$", "$\\Delta F_z$"]
    model.u_labels = ["$a$", "$\\delta$"]
    model.p_labels = ["$s$"]
    return model

def CreateOcpSolver_SpatialDyn_withDeltaFz() -> AcadosOcp:
    ocp = AcadosOcp()
    model = Spatial_DynamicBicycleModel_withDeltaFz()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()

    # keep same outputs as before (lateral/heading errors + vx + controls)
    ny = 3 + nu    # [epsi+beta, e_y, vx] + u
    ny_e = 2       # terminal: [epsi+atan(vy/vx), e_y]

    ocp.solver_options.N_horizon = N_horizon

    Q_mat = np.diag([2e1, 1e1, 1e0])   # weights for [epsi term, e_y, vx]
    R_mat = np.diag([1e-1, 1e-1])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))

    # cost y expression - update indices for vx, vy
    # model.x[0] is e_psi ; model.x[5] is vy ; model.x[4] is vx
    ocp.model.cost_y_expr = vertcat(
        model.x[0] + arctan(model.x[5] / (model.x[4] + 1e-5)),
        model.x[1],
        model.x[4],
        model.u
    )

    # terminal
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    Qe = np.diag([1e1, 5e1])
    ocp.cost.W_e = Qe * ds_ocp
    ocp.cost.yref_e = np.array([0.0, 0.0])
    ocp.model.cost_y_expr_e = vertcat(
        model.x[0],
        model.x[1]
    )

    # parameter init (starting s)
    ocp.parameter_values = np.array([s_ref[0]])

    # input bounds
    ocp.constraints.lbu = np.array([-1.0, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1.0, np.deg2rad(30)])
    ocp.constraints.idxbu = np.array([0, 1])

    # initial condition (note: X0 must have size nx)
    ocp.constraints.x0 = X0  # <-- X0 ensured to have length nx

    # simple bounds on epsi and e_y as before
    ocp.constraints.lbx = np.array([-np.deg2rad(80), -0.5])
    ocp.constraints.ubx = np.array([ np.deg2rad(80), 0.5])
    ocp.constraints.idxbx = np.array([0, 1])

    # solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 70
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = Tf
    return ocp

def time_dynamical_bicycle_model_normalforces():
    # States
    x = SX.sym('x')
    y = SX.sym('y')
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    psi = SX.sym('psi')
    omega = SX.sym('omega')  # yaw rate - psi dot
    delta_Fz = SX.sym('delta_Fz')
    x = vertcat(x, y, vx, vy, psi, omega, delta_Fz)

    # Controls: acceleration a, steering delta
    delta = SX.sym('delta')
    a = SX.sym('a')
    u = vertcat(a, delta)

    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    vx_dot = SX.sym("vx_dot")
    vy_dot = SX.sym("vy_dot")
    psi_dot = SX.sym("psi_dot")
    omega_dot = SX.sym('omega_dot')
    delta_Fz_dot = SX.sym('delta_Fz_dot')
    xdot = vertcat(x_dot, y_dot, vx_dot, vy_dot, psi_dot, omega_dot, delta_Fz_dot)

    vy_f = vy + params.lf * omega
    vy_r = vy - params.lr * omega
    vx_f = vx
    vx_r = vx

    # velocities for each wheel frame
    vc_f = vy_f * cos(delta) - vx_f * sin(delta)
    vc_r = vy_r
    vl_f = vy_f * sin(delta) + vx_f * cos(delta)
    vl_r = vx_r

    beta_f = arctan(vc_f / (vl_f + 1e-5))
    beta_r = arctan(vc_r / (vl_r + 1e-5))

    Fx_d = 0.5 * params.ro * params.Cz * params.Az * vx * fabs(vx)

    Fc_f = -params.mi * params.Dcf * sin(params.Ccf * arctan(1 / params.mi * params.Bcf * beta_f))
    Fc_r = -params.mi * params.Dcr * sin(params.Ccr * arctan(1 / params.mi * params.Bcr * beta_r))

    Fzf_0 = params.m * params.g * params.lr / (params.L)
    Fzr_0 = params.m * params.g * params.lf / (params.L)

    Fzf = Fzf_0 - delta_Fz
    Fzr = Fzr_0 + delta_Fz

    gamma = tanh(a / a_max)  # <-- tanh imported
    Fl_f = gamma * params.mi * Fzf
    Fl_r = gamma * params.mi * Fzr

    Fx_f = Fl_f * cos(delta) - Fc_f * sin(delta)
    Fx_r = Fl_r
    Fx = Fx_f + Fx_r

    Fy_f = Fl_f * sin(delta) + Fc_f * cos(delta)
    Fy_r = Fc_r

    deltaz = -params.h * Fx / (2 * params.L)
    tau_z = 0.5

    dX = vx * cos(psi) - vy * sin(psi)
    dY = vx * sin(psi) + vy * cos(psi)
    dvx = vy * omega + (Fx - Fx_d) / params.m
    dvy = - vx * omega + (Fy_f + Fy_r) / params.m
    dpsi = omega
    domega = (params.lf * Fy_f - params.lr * Fy_r) / params.I_z
    ddelta_Fz = (deltaz - delta_Fz) / tau_z

    f_expl = vertcat(dX, dY, dvx, dvy, dpsi, domega, ddelta_Fz)
    f_impl = xdot - f_expl
    model = AcadosModel()
    model.name = 'time_bicyc_dynamical_model_deltaFz'
    model.x = x
    model.xdot = xdot
    model.u = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model

def create_ocp_solver_BodyFrame1() -> AcadosOcp:
    ocp = AcadosOcp()
    model = time_dynamical_bicycle_model_normalforces()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N_horizon
    Q_mat = np.diag([1e1, 1e1, 1e1, 1e-1, 1e0, 1e-1, 1e-2])  # adjusted length to match nx=7
    R_mat = np.diag([1e-1, 1e-1])
    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.yref = np.zeros((ny,))
    ocp.model.cost_y_expr = vertcat(model.x[:4], model.x[4] + arctan(model.x[3] / (model.x[2] + 1e-5)), model.x[5], model.x[6], model.u)

    # terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.diag([5*1e1, 1e1, 5*1e1, 1e0, 1e-1, 1e-1, 1e-2]) * ds_ocp
    yref_e = np.array([7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.yref_e = yref_e
    ocp.model.cost_y_expr_e = model.x

    # set constraints
    ocp.constraints.lbu = np.array([-a_max, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([a_max, np.deg2rad(30)])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = X0

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
    ocp = CreateOcpSolver_SpatialDyn_withDeltaFz()
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_spatial1.json')

    sim_solver = create_ocp_solver_BodyFrame1()  # returns an OCP object whose .model we will reuse
    sim = AcadosSim()
    sim.model = sim_solver.model
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.T = dt_sim  # [s]
    acados_sim_solver = AcadosSimSolver(sim, json_file='acados_sim_spatial1.json')

    # Simulation settings
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    predX = np.zeros((Nsim + N_horizon + 1, nx))
    predU = np.zeros((Nsim + N_horizon, nu))

    # Initialization
    # time-domain state should match time-model x: [x, y, vx, vy, psi, omega, delta_Fz] (7 elements)
    # y_ref[0, 2:] typically yields [x, y, vx, vy, psi] (5 elements) -> append omega=0 and delta_Fz=0
    xcurrent = np.hstack((y_ref[0, 2:], 0.0))   # <-- ensure length = 7
    xcurr_ocp = X0.copy()    # spatial-domain state [eψ, ey, x, y, vx, vy, ψ, ω, ΔFz]
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
            # Compute current s, epsi, ey from time-domain state
            s_now, epsi_now, ey_now = time2spatial(
                xcurrent[0], xcurrent[1], xcurrent[4], s_ref, y_ref[:, [2, 3, 6]]
            )

            # Find nearest reference index
            idx_closest = int(np.argmin(np.abs(s_ref - s_now)))

            # Update horizon references
            for j in range(N_horizon):
                idx_stage = min(idx_closest + j, len(s_ref) - 1)
                yref_stage = np.concatenate(
                    (y_ref[idx_stage, :2], [y_ref[idx_stage, 4]], [0.0, 0.0])
                )
                acados_ocp_solver.set(j, "yref", yref_stage)
                acados_ocp_solver.set(j, "p", np.array([s_ref[idx_stage]]))

            # Terminal
            idx_term = min(idx_closest + N_horizon, len(s_ref) - 1)
            acados_ocp_solver.set(N_horizon, "yref", y_ref[idx_term, :2])
            acados_ocp_solver.set(N_horizon, "p", np.array([s_ref[idx_term]]))

            # Set initial state constraints for OCP
            x0_ocp = np.hstack((epsi_now, ey_now, xcurrent))  # length 9
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
            xcurrent[0], xcurrent[1], xcurrent[4], s_ref, y_ref[:, [2, 3, 6]]
        )
        simX[i + 1, :] = np.hstack((epsi, ey, xcurrent))
        simX[i + 1, 0] += np.arctan(simX[i + 1, 5] / (simX[i + 1, 4] + 1e-8))
        simX[i + 1, -2] += np.arctan(simX[i + 1, 5] / (simX[i + 1, 4] + 1e-8))
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
    axs[1].legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    # --- 2. States ---
    state_indices = [2, 3, 4, 5, 6, 7]  # x, y, vx, vy, theta, omega
    state_labels = [r'$x[m]$', r'$y[m]$', r'$v_x[\frac{m}{s}]$', r'$v_y[\frac{m}{s}]$', r'$\theta[rad]$', r'$\omega[\frac{rad}{s}]$', r'$\Delta F_z[\frac{rad}{s}]$']
    fig2, ax2 = plt.subplots(len(state_indices) +1, 1, sharex=True, figsize=(6, 12))
    for j, idx in enumerate(state_indices):
        ax2[j].plot(timestampsx, simX[:Nf+1, idx],label = r'Simulation', color="C0", linewidth=2)
        ax2[j].plot(timestampsx, y_ref[:Nf+1, idx], '--',  label = r'Reference', color="#ffd166", linewidth=1.5)
        ax2[j].set_ylabel(state_labels[j])
        ax2[j].grid(True, linestyle="--", alpha=0.4)
    ax2[0].set_title("Position Coordinates", fontsize=12, fontweight='bold')
    ax2[2].set_title("Speed", fontsize=12, fontweight='bold')
    ax2[4].set_title("Direction Angle", fontsize=12, fontweight='bold')
    ax2[6].set_title("Dynamical Vertical Load", fontsize=12, fontweight='bold')
    ax2[6].plot(timestampsx, simX[:Nf+1, 8],label = r'Simulation', color="C0", linewidth=2)
    ax2[6].set_ylabel(state_labels[6])
    ax2[-1].set_xlabel(r'Time$[s]$')
    ax2[-2].legend(fontsize=11, loc='upper right')
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
