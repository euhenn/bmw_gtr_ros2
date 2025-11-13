import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from casadi import SX, vertcat, sin, cos, tan, arctan, interpolant

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosModel

# == your imports ==
from ref_time2spatial import *
from reference_generation_v import *

# ==========================================================
#                      CONFIGURATION
# ==========================================================

lf = 0.1335
lr = 0.1335
L = lf + lr

ds_ocp = 0.01
dt_sim = 0.005
N_horizon = 50
dt_control = 0.02

Tf =N_horizon *ds_ocp
Tf_sim = N_horizon *dt_sim

nodes_to_visit = [141, 91, 125, 150]

traj_gen = TrajectoryGeneration(
    ds=ds_ocp, N_horizon=N_horizon,
    v_max=0.6, v_min=0.3,
    use_curvature_velocity=False, smooth_velocity=True
)

y_ref, s_ref, kappa_ref = traj_gen.generating_spatial_reference(nodes_to_visit)
y_ref = y_ref.T

kappa = interpolant("kappa", "bspline", [s_ref], kappa_ref)

N = len(s_ref) - N_horizon
S_total = N * ds_ocp
Nsim = int(2 * S_total / dt_sim)

X0 = y_ref[0, :]    # correct dimensions already


# ==========================================================
#                 MODEL DEFINITIONS
# ==========================================================

def spatial_bicycle_model():
    model_name = "SpatialBicycleOCP"

    s = SX.sym('s')
    epsi = SX.sym('e_psi')
    ey = SX.sym('e_y')
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v = SX.sym('v')
    xvec = vertcat(epsi, ey, x, y, psi, v)

    a = SX.sym("a")
    delta = SX.sym("delta")
    uvec = vertcat(a, delta)

    epsi_dot = SX.sym("e_psi_dot")
    ey_dot = SX.sym("e_y_dot")
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    psi_dot = SX.sym("psi_dot")
    v_dot = SX.sym("v_dot")

    xdot = vertcat(epsi_dot, ey_dot, x_dot, y_dot, psi_dot, v_dot)

    beta = arctan(lr * tan(delta) / L)
    dpsi = v * sin(beta) / lr

    sdot = (v * cos(beta) * cos(epsi) - v * sin(beta) * sin(epsi)) / (1 - kappa(s) * ey)

    d_epsi = dpsi / sdot - kappa(s)
    d_ey = (v * cos(beta) * sin(epsi) + v * sin(beta) * cos(epsi)) / sdot
    dx = v * cos(psi + beta) / sdot
    dy = v * sin(psi + beta) / sdot
    dpsi_s = dpsi / sdot
    dv = a / sdot

    f_expl = vertcat(d_epsi, d_ey, dx, dy, dpsi_s, dv)
    f_impl = xdot - f_expl

    from acados_template import AcadosModel
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = xvec
    model.xdot = xdot
    model.u = uvec
    model.p = vertcat(s)
    model.name = model_name

    return model


def time_bicycle_model():
    model_name = "TimeBicycleSim"

    x = SX.sym("x")
    y = SX.sym("y")
    psi = SX.sym("psi")
    v = SX.sym("v")
    xvec = vertcat(x, y, psi, v)

    a = SX.sym("a")
    delta = SX.sym("delta")
    uvec = vertcat(a, delta)

    beta = arctan(lr * tan(delta) / L)
    f_expl = vertcat(
        v * cos(psi + beta),
        v * sin(psi + beta),
        v * sin(beta) / lr,
        a
    )
    xdot = SX.sym("xdot", 4)

    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x = xvec
    model.xdot = xdot
    model.u = uvec
    model.name = model_name

    return model


# ==========================================================
#                 OCP + SIM SOLVER BUILDERS
# ==========================================================

def create_ocp_solver():
    ocp = AcadosOcp()
    model = spatial_bicycle_model()

    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    Q = np.diag([100, 50])
    R = np.diag([1, 0.01])

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.model.cost_y_expr = vertcat(model.x[0], model.x[1]+np.arctan(lr*tan(model.u[1])/L), model.u)

    ocp.cost.yref = np.zeros(2 + nu)

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.diag([1000, 500])*ds_ocp
    ocp.model.cost_y_expr_e = model.x[:2]
    ocp.cost.yref_e = np.array([0, 0])

    ocp.constraints.lbu = np.array([-1, -np.deg2rad(30)])
    ocp.constraints.ubu = np.array([1, np.deg2rad(30)])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-np.deg2rad(40), -0.5])
    ocp.constraints.ubx = np.array([np.deg2rad(40), 0.5])
    ocp.constraints.idxbx = np.array([0, 1])

    ocp.constraints.x0 = X0
    ocp.parameter_values = np.array([s_ref[0]])

    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    ocp.solver_options.nlp_solver_max_iter = 70
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.N_horizon = N_horizon

    return ocp


def create_sim_solver():
    ocp = AcadosOcp()
    model = time_bicycle_model()
    ocp.model = model

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.num_stages = 4
    ocp.solver_options.num_steps = 1
    ocp.solver_options.tf = Tf_sim
    ocp.solver_options.T = dt_sim

    return ocp


# ==========================================================
#                    PLOT MANAGER CLASS
# ==========================================================

class PlotManager:
    def reference_at_s(self, y_ref, s_query, s_ref):
        Y = np.zeros((len(s_query), y_ref.shape[1]))
        for j in range(y_ref.shape[1]):
            Y[:, j] = np.interp(s_query, s_ref, y_ref[:, j])
        return Y

    def plot_space(self, simX, simU, S, y_ref, s_ref):

        y_ref_s = self.reference_at_s(y_ref, S, s_ref)

        plt.figure(figsize=(7, 5))
        plt.plot(simX[:, 2], simX[:, 3], lw=2)
        plt.plot(y_ref_s[:, 2], y_ref_s[:, 3], '--', lw=1.5)
        plt.axis("equal")
        plt.grid(True)
        plt.title("Path Tracking (Spatial Domain)")

        fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        ax[0].plot(S, simX[:, 1], lw=2)
        ax[0].plot(S, y_ref_s[:, 1], '--', lw=1.5)
        ax[0].set_ylabel("e_y")

        ax[1].plot(S, simX[:, 0], lw=2)
        ax[1].plot(S, y_ref_s[:, 0], '--', lw=1.5)
        ax[1].set_ylabel("e_psi")
        ax[1].set_xlabel("Arc length s [m]")
        ax[1].grid(True)

        plt.show()


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


# ==========================================================
#                CLOSED LOOP SIMULATION
# ==========================================================

def closed_loop_simulation():

    # Build solvers
    ocp = create_ocp_solver()
    solver_ocp = AcadosOcpSolver(ocp, json_file="ocp.json")

    sim_ocp = create_sim_solver()
    sim_solver = AcadosSimSolver(sim_ocp, json_file="sim.json")

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    S_sim = np.zeros(Nsim + 1)

    xcur = y_ref[0, 2:]
    simX[0] = X0

    plotter = PlotManager()
    saver = DataLogger()

    # ================ MAIN LOOP ======================
    control_timer = 0.0

    for i in range(Nsim):

        if control_timer >= dt_control:

            # compute spatial error from time states
            s_now, epsi_now, ey_now = time2spatial(
                xcur[0], xcur[1], xcur[2],
                s_ref, y_ref[:, 2:5]
            )

            idx = np.argmin(np.abs(s_ref - s_now))

            x0 = np.hstack((epsi_now, ey_now, xcur))
            solver_ocp.set(0, "lbx", x0)
            solver_ocp.set(0, "ubx", x0)

            for k in range(N_horizon):
                j = min(idx + k, len(s_ref) - 1)
                solver_ocp.set(k, "p", np.array([s_ref[j]]))
                solver_ocp.set(k, "yref", np.array([y_ref[j, 0], y_ref[j, 1], 0, 0]))

            solver_ocp.solve()
            u0 = solver_ocp.get(0, "u")

            control_timer -= dt_control
        else:
            u0 = np.zeros(nu)

        # === SIMULATE ===
        simU[i] = u0
        xcur = sim_solver.simulate(xcur, u0)
        

        # compute new s
        s_now, epsi_sim, ey_sim = time2spatial(
            xcur[0], xcur[1], xcur[2],
            s_ref, y_ref[:, 2:5]
        )
        S_sim[i + 1] = s_now
        simX[i + 1] = np.hstack((epsi_sim, ey_sim, xcur))

        control_timer += dt_sim

        if s_now >= S_total:
            break

    # ======= FINALIZE =======
    S_sim = S_sim[: i + 2]
    simX = simX[: i + 2]
    simU = simU[: i + 1]

    y_ref_s = plotter.reference_at_s(y_ref, S_sim, s_ref)

    plotter.plot_space(simX, simU, S_sim, y_ref, s_ref)
    saver.save_csv(simX, simU, S_sim, y_ref_s)

    print("Simulation finished.")


if __name__ == "__main__":
    closed_loop_simulation()
