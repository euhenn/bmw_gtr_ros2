from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import scipy.linalg
from casadi import SX, vertcat, cos, sin, tan, arctan, interpolant, fmax
from reference_generation_velocity import TrajectoryGeneration

DRIVING_MODE = 'aggressive'  # options: 'precision', 'aggressive', 'balanced', 'emergency'

class VehicleParameters:
    """Container for vehicle physical parameters"""
    def __init__(self):
        self.lf = 0.13  # distance from CoG to front wheels [m]
        self.lr = 0.13  # distance from CoG to rear wheels [m]
        self.L = self.lf + self.lr  # wheelbase [m]


class MPCParameters:
    """Container for MPC tuning parameters"""
    def __init__(self, ds=0.06, N_horizon=50):
        self.ds = ds  # spatial step [m]
        self.N_horizon = N_horizon  # prediction horizon
        self.Tf = N_horizon * ds  # total horizon length
        
        # Cost function weights
        self.Q_weights = np.array([5e3, 1e2, 1e-2])  # state weights [e_psi, e_y, v]
        self.Q_weights = np.array([5e3, 1e2, 1e0])  # Increase velocity weight significantly
        self.R_weights = np.array([1e2, 1e-2])  # control weights [a, delta]

        if DRIVING_MODE == 'precision':
            #Low-Speed Precision Tracking
            self.Q_weights = np.array([1e1, 1e1, 1e0]) 
            self.R_weights = np.array([5e-2, 1e-2])        
        elif DRIVING_MODE == 'aggressive': 
            #Aggressive Racing / Tight Track
            self.Q_weights = np.array([1e3, 1e2, 5e-1]) 
            self.R_weights = np.array([7e-2, 1e-2])      
        elif DRIVING_MODE == 'balanced':
            #balanced driving
            self.Q_weights = np.array([1e4, 1e-1, 5e-1]) 
            self.R_weights = np.array([7e-2, 1e-2])     
        elif DRIVING_MODE == 'emergency':
            #Emergency Maneuver
            self.Q_weights = np.array([1e3, 1e2, 5e-1]) 
            self.R_weights = np.array([7e-2, 1e-2])      
        
        # Constraints
        self.a_min, self.a_max = -1.0, 1.0  # acceleration limits [m/s²]
        self.delta_min, self.delta_max = np.deg2rad(-28), np.deg2rad(28)  # steering limits [rad]
        self.e_psi_min, self.e_psi_max = np.deg2rad(-100), np.deg2rad(100)  # heading error limits [rad]
        self.e_y_min, self.e_y_max = -1.20, 1.20  # lateral error limits [m]


class KinematicBicycleModel:
    """Spatial kinematic bicycle model for path tracking"""
    
    def __init__(self, vehicle_params):
        self.vehicle = vehicle_params
        
    def create_model(self, kappa_interpolant):
        """Create Acados model with spatial dynamics"""
        # State variables: [e_psi, e_y, x, y, psi, v]
        e_psi = SX.sym('e_psi')  # heading error
        e_y = SX.sym('e_y')      # lateral error
        x = SX.sym('x')          # global x position
        y = SX.sym('y')          # global y position
        psi = SX.sym('psi')      # vehicle heading
        v = SX.sym('v')          # velocity
        states = vertcat(e_psi, e_y, x, y, psi, v)
        
        # Control inputs: [a, delta]
        a = SX.sym('a')          # acceleration
        delta = SX.sym('delta')  # steering angle
        controls = vertcat(a, delta)
        
        # State derivatives
        states_dot = SX.sym('states_dot', states.shape)
        
        # Model parameters: path curvature at current position
        s = SX.sym('s')  # path coordinate
        parameters = vertcat(s)
        
        # Vehicle kinematics
        beta = arctan(self.vehicle.lr * tan(delta) / self.vehicle.L)  # sideslip angle
        vx = v * cos(psi + beta)  # longitudinal velocity
        vy = v * sin(psi + beta)  # lateral velocity
        dpsi = v * sin(beta) / self.vehicle.lr  # yaw rate
        
        # Spatial dynamics: dx/ds = f(x,u)
        s_dot = (v * cos(beta) * cos(e_psi) - v * sin(beta) * sin(e_psi)) / (1 - kappa_interpolant(s) * e_y)
        
        # Avoid division by near-zero values
        #denominator = 1 - kappa_interpolant(s) * e_y
        #s_dot = (v * cos(beta) * cos(e_psi) - v * sin(beta) * sin(e_psi)) / fmax(denominator, 0.01)
        
        de_psi_ds = dpsi / s_dot - kappa_interpolant(s)  # heading error dynamics
        de_y_ds = (v * cos(beta) * sin(e_psi) + v * sin(beta) * cos(e_psi)) / s_dot  # lateral error dynamics
        dx_ds = vx / s_dot      # x position dynamics
        dy_ds = vy / s_dot      # y position dynamics
        dpsi_ds = dpsi / s_dot  # heading dynamics
        dv_ds = a / s_dot       # velocity dynamics
        
        dynamics_explicit = vertcat(de_psi_ds, de_y_ds, dx_ds, dy_ds, dpsi_ds, dv_ds)
        dynamics_implicit = states_dot - dynamics_explicit
        
        # Create Acados model
        model = AcadosModel()
        model.f_impl_expr = dynamics_implicit
        model.f_expl_expr = dynamics_explicit
        model.x = states
        model.xdot = states_dot
        model.u = controls
        model.p = parameters
        
        # Labels for debugging/visualization
        model.x_labels = ["e_psi", "e_y", "x", "y", "psi", "v"]
        model.u_labels = ["a", "delta"]
        model.p_labels = ["s"]
        model.name = "SpatialKinematicBicycle_model"
        
        return model


class PathTrackerMPC:
    """MPC controller for path tracking using spatial formulation"""
    
    def __init__(self, ds=0.06, N_horizon=50, trajectory_nodes=None):
        if trajectory_nodes is None:
            trajectory_nodes = [73, 97, 125]
            
        self.vehicle_params = VehicleParameters()
        self.mpc_params = MPCParameters(ds, N_horizon)
        self.model = KinematicBicycleModel(self.vehicle_params)
        
        # Generate reference trajectory
        self._generate_reference_trajectory(trajectory_nodes)
        
        # Setup MPC solver
        self._setup_mpc_solver()
    
    def _generate_reference_trajectory(self, nodes):
        """Generate reference path and curvature profile"""
        self.trajectory_generator = TrajectoryGeneration(self.mpc_params.ds, self.mpc_params.N_horizon,v_max=0.6, v_min=0.2, use_curvature_velocity=True)

        self.trajectory, self.s_ref, kappa_ref = self.trajectory_generator.generating_spatial_reference(nodes)
        
        # Create curvature interpolant for spatial dynamics
        self.kappa = interpolant("kappa", "bspline", [self.s_ref], kappa_ref)
        self.initial_state = self.trajectory[:, 0]
    
    def _setup_mpc_solver(self):
        """Configure and create Acados OCP solver"""
        ocp = AcadosOcp()

        model = self.model.create_model(self.kappa)
        ocp.model = model
        self._setup_cost_function(ocp)
        self._setup_constraints(ocp)
        self._setup_solver_options(ocp)
        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    
    def _setup_cost_function(self, ocp):
        """Configure MPC cost function"""
        nx, nu = ocp.model.x.rows(), ocp.model.u.rows()
        
        # Stage cost: nonlinear least squares formulation
        ocp.cost.cost_type = "NONLINEAR_LS"
        Q_mat = np.diag(self.mpc_params.Q_weights)
        R_mat = np.diag(self.mpc_params.R_weights)
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.yref = np.zeros((3 + nu,))  # [e_psi + beta, e_y, v, u]
        
        # Cost expression: track path errors and velocity
        beta_expr = arctan(self.vehicle_params.lr * tan(ocp.model.u[1]) / self.vehicle_params.L)
        ocp.model.cost_y_expr = vertcat(
            ocp.model.x[0] + beta_expr,  # heading error with lookahead
            ocp.model.x[1],              # lateral error
            ocp.model.x[-1],             # velocity
            ocp.model.u                  # controls
        )
        
        # Terminal cost
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag(self.mpc_params.Q_weights[:2]) * self.mpc_params.ds
        ocp.cost.yref_e = np.zeros(2)  # terminal path errors
        ocp.model.cost_y_expr_e = vertcat(ocp.model.x[:2])  # terminal state cost
    
    def _setup_constraints(self, ocp):
        """Setup state and control constraints"""
        # Control constraints
        ocp.constraints.lbu = np.array([self.mpc_params.a_min, self.mpc_params.delta_min])
        ocp.constraints.ubu = np.array([self.mpc_params.a_max, self.mpc_params.delta_max])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # State constraints (path errors)
        ocp.constraints.lbx = np.array([self.mpc_params.e_psi_min, self.mpc_params.e_y_min])
        ocp.constraints.ubx = np.array([self.mpc_params.e_psi_max, self.mpc_params.e_y_max])
        ocp.constraints.idxbx = np.array([0, 1])
        
        # Initial state
        ocp.constraints.x0 = self.initial_state
        ocp.parameter_values = np.array([self.s_ref[0]])
    
    def _setup_solver_options(self, ocp):
        """Configure Acados solver options"""
        ocp.solver_options.N_horizon = self.mpc_params.N_horizon
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = self.mpc_params.Tf
    
    def set_initial_state(self, x0):
        """Set initial state constraint"""
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
    
    def set_reference(self, current_idx):
        """Set reference trajectory for MPC horizon"""
        if current_idx + self.mpc_params.N_horizon >= len(self.s_ref):
            raise ValueError("Reference trajectory too short for prediction horizon")
        
        # Set stage references
        for j in range(self.mpc_params.N_horizon):
            ref_idx = current_idx + j
            y_ref = np.concatenate([
                self.trajectory[:2, ref_idx],  # path errors
                [self.trajectory[-1, ref_idx]],  # velocity
                np.zeros(2)  # zero control reference
            ])
            self.solver.set(j, "yref", y_ref)
            self.solver.set(j, "p", np.array([self.s_ref[ref_idx]]))
        
        # Set terminal reference
        terminal_idx = current_idx + self.mpc_params.N_horizon
        self.solver.set(self.mpc_params.N_horizon, "yref", self.trajectory[:2, terminal_idx])
        self.solver.set(self.mpc_params.N_horizon, "p", np.array([self.s_ref[terminal_idx]]))
    
    def solve(self, current_state, current_idx, warm_start_u=None):
        """
        Solve MPC optimization problem
        
        Args:
            current_state: current vehicle state [e_psi, e_y, x, y, psi, v]
            current_idx: current index along reference trajectory
            warm_start_u: optional warm start controls [a, delta]
        
        Returns:
            tuple: (acceleration_cmd, steering_cmd)
        """
        self.set_initial_state(current_state)
        self.set_reference(current_idx)
        
        # Warm start with previous controls if provided
        if warm_start_u is not None:
            a_prev, delta_prev = warm_start_u
            for j in range(self.mpc_params.N_horizon):
                try:
                    self.solver.set(j, "u", np.array([a_prev, delta_prev]))
                except Exception:
                    pass  # Warm start might fail for some stages
        
        # Solve OCP
        status = self.solver.solve()
        
        if status not in [0, 2]:  # 0: SUCCESS, 2: MAX_ITER
            raise RuntimeError(f"MPC solve failed with status {status}")
        
        # Extract first control input
        u_opt = self.solver.get(0, "u")
        return u_opt[0], u_opt[1]  # a_cmd, delta_cmd


# Backward compatibility
MPC_KinematicBicycle = PathTrackerMPC