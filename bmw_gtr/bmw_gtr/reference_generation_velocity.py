"""
Trajectory Generation for MPC Controller WITH CURVATURE-ADAPTIVE VELOCITY PROFILE

Generates spatial reference trajectories from waypoints with:
- Position (x, y), heading angle (theta)
- Velocity profile (constant or curvature-adaptive)
- Curvature information
- Online horizon generation for MPC

Outputs:
- trajectory: [e_theta, e_y, x, y, theta, v] (6 x N)
- s_uniform: arc-length parameterization
- kappa: curvature profile
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import cv2 as cv
from path_planning_eugen import PathPlanning 


class TrajectoryGeneration:
    def __init__(self, ds, N_horizon, v_max=0.4, v_min=0.1, use_curvature_velocity=False):
        """
        Initialize trajectory generator
        
        Args:
            ds: Spatial sampling distance [m]
            N_horizon: MPC prediction horizon length
            v_max: Maximum velocity [m/s]
            v_min: Minimum velocity [m/s] (used when use_curvature_velocity=True)
            use_curvature_velocity: If True, adapt velocity based on curvature
        """
        #track = cv.imread('data/final_map.png')
        track = cv.imread('data/2024_VerySmall.png')
        self.planner = PathPlanning(track)
        self.N_horizon = N_horizon
        self.ds = ds
        
        # Velocity profile parameters
        self.v_max = v_max
        self.v_min = v_min
        self.use_curvature_velocity = use_curvature_velocity
        self.k_threshold = 0.5  # Curvature threshold [1/m] for velocity adaptation
        
        # Initialize splines and path length
        self.cs_x = None
        self.cs_y = None
        self.S_length = None
        self.nodes_to_visit = None
        self.kappa_full = None  # Store full curvature for online reference

    def _compute_velocity_profile(self, kappa):
        """
        Compute velocity profile based on curvature
        
        Args:
            kappa: Curvature array [1/m]
            
        Returns:
            v: Velocity profile [m/s]
        """
        if self.use_curvature_velocity:
            # Slow down in high curvature regions
            # Linear interpolation between v_max (straight) and v_min (high curvature)
            curvature_factor = np.clip(np.abs(kappa) / self.k_threshold, 0, 1)
            v = self.v_max - (self.v_max - self.v_min) * curvature_factor
        else:
            # Constant velocity profile
            v = self.v_max * np.ones_like(kappa)
        
        return v

    def _generate_dense_spatial_ref(self):
        """Internal method to create dense path and build splines"""
        # Generate dense path through waypoints
        self.planner.generate_path_passing_through(self.nodes_to_visit, step_length=0.01, method='spline')


        ####--------------------- REMOVE COMMENTED TO PLOT THE PATH WHEN RUNNING THE MPC -------------------------------------------####
        self.planner.plot_path_and_curvature(show_map=True)


        route = self.planner.route_list  # [(x0,y0,yaw0), (x1,y1,yaw1), ...]

        # Extract X and Y
        X = np.array([p[0] for p in route])
        Y = np.array([p[1] for p in route])

        # Parameterize path by cumulative arc length
        s = np.zeros(len(X))
        s[1:] = np.cumsum(np.hypot(np.diff(X), np.diff(Y)))

        # Build cubic splines X(s), Y(s)
        self.cs_x = CubicSpline(s, X)
        self.cs_y = CubicSpline(s, Y)
        self.S_length  = s[-1]

        return self.cs_x, self.cs_y, self.S_length

    def generating_spatial_reference(self, nodes_to_visit):
        """
        Generate complete spatial reference trajectory
        
        Args:
            nodes_to_visit: List of waypoint indices to pass through
            
        Returns:
            trajectory: (6, N) array [e_theta, e_y, x, y, theta, v]
            s_uniform: Arc-length values [m]
            kappa: Curvature profile [1/m]
        """
        self.nodes_to_visit = nodes_to_visit
        cs_x, cs_y, S_length = self._generate_dense_spatial_ref()
        
        # Create uniform arc-length grid
        s_uniform = np.arange(0.0, S_length, self.ds)
        if s_uniform[-1] < S_length:
            s_uniform = np.append(s_uniform, S_length)

        # Evaluate spline at uniform s
        Xu = cs_x(s_uniform)
        Yu = cs_y(s_uniform)
        
        # Initialize errors (zero for reference trajectory)
        e_theta = np.zeros_like(Xu)
        e_y = np.zeros_like(Xu)

        # Compute derivatives for heading and curvature
        dx_ds = cs_x.derivative(1)(s_uniform)
        dy_ds = cs_y.derivative(1)(s_uniform)
        ddx_ds2 = cs_x.derivative(2)(s_uniform)
        ddy_ds2 = cs_y.derivative(2)(s_uniform)

        # Heading angle (unwrap to avoid discontinuities)
        theta = np.arctan2(dy_ds, dx_ds)
        theta = np.unwrap(theta)
        
        # Curvature calculation with numerical stability
        denom = np.hypot(dx_ds, dy_ds)**3
        denom = np.maximum(denom, 1e-9)  # Avoid division by zero
        kappa = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom
        kappa = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / (dx_ds**2 + dy_ds**2)**1.5
        
        # Store curvature for online reference generation
        self.kappa_full = kappa
        
        # Compute velocity profile (constant or curvature-adaptive)
        v = self._compute_velocity_profile(kappa)

        # Stack into trajectory format
        trajectory = np.stack((e_theta, e_y, Xu, Yu, theta, v)).astype(np.float32)

        return trajectory, s_uniform, kappa

    def generating_online_spatial_ref(self, s):
        """
        Generate MPC prediction horizon starting from arc-length s
        
        Args:
            s: Current arc-length position [m]
            
        Returns:
            trajectory: (6, N_horizon) array for MPC reference
        """
        if self.cs_x is None or self.cs_y is None:
            raise RuntimeError("Must call generating_spatial_reference first")
        
        # Clip to valid range
        s = np.clip(s, 0, self.S_length)
        
        # Create horizon arc-length points
        s_end = s + (self.N_horizon - 1) * self.ds
        s_horizon = np.linspace(s, s_end, self.N_horizon)
        s_horizon = np.clip(s_horizon, 0, self.S_length)
        
        # Evaluate splines
        Xu = self.cs_x(s_horizon)
        Yu = self.cs_y(s_horizon)
        e_theta = np.zeros_like(Xu)
        e_y = np.zeros_like(Xu)

        # Compute heading and curvature
        dx_ds = self.cs_x.derivative(1)(s_horizon)
        dy_ds = self.cs_y.derivative(1)(s_horizon)
        ddx_ds2 = self.cs_x.derivative(2)(s_horizon)
        ddy_ds2 = self.cs_y.derivative(2)(s_horizon)
        
        theta = np.arctan2(dy_ds, dx_ds)
        theta = np.unwrap(theta)
        
        # Compute curvature for velocity profile
        denom = np.hypot(dx_ds, dy_ds)**3
        denom = np.maximum(denom, 1e-9)
        kappa_horizon = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom
        
        # Compute velocity profile (consistent with full reference)
        v = self._compute_velocity_profile(kappa_horizon)

        trajectory = np.stack((e_theta, e_y, Xu, Yu, theta, v)).astype(np.float32)

        return trajectory


def plot_trajectory_analysis(trajectory, s_uniform, kappa, nodes, ds, track_gen):
    """Create comprehensive visualization of trajectory"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Trajectory Generation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spatial trajectory
    axes[0,0].plot(trajectory[2, :], trajectory[3, :], 'b-', linewidth=2)
    axes[0,0].plot(trajectory[2, 0], trajectory[3, 0], 'go', markersize=10, 
                   label='Start', markeredgecolor='black')
    axes[0,0].plot(trajectory[2, -1], trajectory[3, -1], 'ro', markersize=10, 
                   label='End', markeredgecolor='black')
    axes[0,0].set_xlabel('X [m]')
    axes[0,0].set_ylabel('Y [m]')
    axes[0,0].set_title('Spatial Trajectory')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axis('equal')
    
    # 2. Heading angle
    axes[0,1].plot(s_uniform, np.degrees(trajectory[4, :]), 'g-', linewidth=2)
    axes[0,1].set_xlabel('Arc Length [m]')
    axes[0,1].set_ylabel('Heading [degrees]')
    axes[0,1].set_title('Heading Angle')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Curvature
    axes[0,2].plot(s_uniform, kappa, 'r-', linewidth=2)
    axes[0,2].set_xlabel('Arc Length [m]')
    axes[0,2].set_ylabel('Curvature [1/m]')
    axes[0,2].set_title('Path Curvature')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Velocity profile
    axes[1,0].plot(s_uniform, trajectory[5, :], 'purple', linewidth=2)
    axes[1,0].set_xlabel('Arc Length [m]')
    axes[1,0].set_ylabel('Velocity [m/s]')
    axes[1,0].set_title('Velocity Profile')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Errors (should be zero)
    axes[1,1].plot(s_uniform, trajectory[0, :], 'orange', linewidth=2, 
                   label='Heading error')
    axes[1,1].plot(s_uniform, trajectory[1, :], 'brown', linewidth=2, 
                   label='Cross-track error')
    axes[1,1].set_xlabel('Arc Length [m]')
    axes[1,1].set_ylabel('Error [rad/m]')
    axes[1,1].set_title('Reference Errors (Should be Zero)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Online horizon generation
    s_test = [0.5, s_uniform[-1]/3, 2*s_uniform[-1]/3]
    colors = ['red', 'blue', 'green']
    
    axes[1,2].plot(trajectory[2, :], trajectory[3, :], 'k-', alpha=0.3, 
                   label='Full trajectory')
    
    for i, s_current in enumerate(s_test):
        if s_current < s_uniform[-1]:
            online_traj = track_gen.generating_online_spatial_ref(s_current)
            axes[1,2].plot(online_traj[2, :], online_traj[3, :], 
                          color=colors[i], linewidth=3, 
                          label=f'Horizon @ s={s_current:.1f}m')
            axes[1,2].plot(online_traj[2, 0], online_traj[3, 0], 
                          'o', color=colors[i], markersize=8, markeredgecolor='black')
    
    axes[1,2].set_xlabel('X [m]')
    axes[1,2].set_ylabel('Y [m]')
    axes[1,2].set_title('Online Reference Generation')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].axis('equal')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    N_horizon = 50
    ds = 0.06
    
    # Test both velocity modes
    print("="*60)
    print("TESTING CONSTANT VELOCITY MODE")
    print("="*60)
    
    track_constant = TrajectoryGeneration(
        ds=ds, 
        N_horizon=N_horizon,
        v_max=0.5,
        use_curvature_velocity=False
    )
    
    nodes = [73, 97, 125]
    trajectory_const, s_uniform, kappa = track_constant.generating_spatial_reference(nodes)
    
    print(f"\nTrajectory Information:")
    print(f"  Shape: {trajectory_const.shape}")
    print(f"  Path length: {s_uniform[-1]:.2f} m")
    print(f"  Number of points: {len(s_uniform)}")
    print(f"  Velocity: constant {trajectory_const[5, 0]:.2f} m/s")
    
    print("\n" + "="*60)
    print("TESTING CURVATURE-ADAPTIVE VELOCITY MODE")
    print("="*60)
    
    track_adaptive = TrajectoryGeneration(
        ds=ds,
        N_horizon=N_horizon,
        v_max=0.5,
        v_min=0.2,
        use_curvature_velocity=True
    )
    
    trajectory_adapt, s_uniform, kappa = track_adaptive.generating_spatial_reference(nodes)
    
    print(f"\nTrajectory Information:")
    print(f"  Shape: {trajectory_adapt.shape}")
    print(f"  Velocity range: {trajectory_adapt[5, :].min():.2f} - {trajectory_adapt[5, :].max():.2f} m/s")
    print(f"  Mean velocity: {trajectory_adapt[5, :].mean():.2f} m/s")
    
    print(f"\nPath Statistics:")
    print(f"  Max curvature: {np.max(np.abs(kappa)):.3f} 1/m")
    print(f"  Max heading change: {np.degrees(np.ptp(trajectory_adapt[4, :])):.1f}°")
    
    # Compare velocity profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Velocity Profile Comparison', fontsize=16, fontweight='bold')
    
    # Constant velocity
    axes[0, 0].plot(s_uniform, trajectory_const[5, :], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Arc Length [m]')
    axes[0, 0].set_ylabel('Velocity [m/s]')
    axes[0, 0].set_title('Constant Velocity Profile')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 2])
    
    # Curvature-adaptive velocity
    axes[0, 1].plot(s_uniform, trajectory_adapt[5, :], 'g-', linewidth=2, label='Velocity')
    ax_curv = axes[0, 1].twinx()
    ax_curv.plot(s_uniform, np.abs(kappa), 'r--', alpha=0.5, label='|Curvature|')
    axes[0, 1].set_xlabel('Arc Length [m]')
    axes[0, 1].set_ylabel('Velocity [m/s]', color='g')
    ax_curv.set_ylabel('|Curvature| [1/m]', color='r')
    axes[0, 1].set_title('Curvature-Adaptive Velocity Profile')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 2])
    axes[0, 1].tick_params(axis='y', labelcolor='g')
    ax_curv.tick_params(axis='y', labelcolor='r')
    
    # Spatial comparison
    axes[1, 0].plot(trajectory_const[2, :], trajectory_const[3, :], 'b-', 
                    linewidth=2, label='Path')
    scatter = axes[1, 0].scatter(trajectory_const[2, :], trajectory_const[3, :], 
                                 c=trajectory_const[5, :], cmap='viridis', 
                                 s=10, alpha=0.6)
    axes[1, 0].set_xlabel('X [m]')
    axes[1, 0].set_ylabel('Y [m]')
    axes[1, 0].set_title('Constant Velocity (colored by speed)')
    axes[1, 0].axis('equal')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Velocity [m/s]')
    
    axes[1, 1].plot(trajectory_adapt[2, :], trajectory_adapt[3, :], 'g-', 
                    linewidth=2, label='Path')
    scatter = axes[1, 1].scatter(trajectory_adapt[2, :], trajectory_adapt[3, :], 
                                 c=trajectory_adapt[5, :], cmap='viridis', 
                                 s=10, alpha=0.6)
    axes[1, 1].set_xlabel('X [m]')
    axes[1, 1].set_ylabel('Y [m]')
    axes[1, 1].set_title('Curvature-Adaptive Velocity (colored by speed)')
    axes[1, 1].axis('equal')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Velocity [m/s]')
    
    plt.tight_layout()
    plt.show()
    
    # Test online reference consistency
    print("\n" + "="*60)
    print("TESTING ONLINE REFERENCE CONSISTENCY")
    print("="*60)
    
    test_s = 1.5
    online_ref_const = track_constant.generating_online_spatial_ref(test_s)
    online_ref_adapt = track_adaptive.generating_online_spatial_ref(test_s)
    
    print(f"\nConstant velocity mode at s={test_s:.1f}m:")
    print(f"  Horizon velocity: {online_ref_const[5, 0]:.2f} m/s (should match reference)")
    
    print(f"\nAdaptive velocity mode at s={test_s:.1f}m:")
    print(f"  Horizon velocity range: {online_ref_adapt[5, :].min():.2f} - {online_ref_adapt[5, :].max():.2f} m/s")
    
    # Full visualization for adaptive mode
    #print("\n" + "="*60)
    #print("GENERATING FULL ANALYSIS PLOTS (Adaptive Mode)")
    #print("="*60)
    #plot_trajectory_analysis(trajectory_adapt, s_uniform, kappa, nodes, ds, track_adaptive)