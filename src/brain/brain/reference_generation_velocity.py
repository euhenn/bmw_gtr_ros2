#!/usr/bin/python3
"""
Trajectory Generation for MPC Controller
-----------------------------------------
Generates spatial reference trajectories from `PathPlanning` class with:
- Position (x, y)
- Heading (psi)
- Curvature (kappa)
- Smoothed velocity profile (constant or curvature-adaptive)
- Online horizon generation for MPC

Outputs:
- trajectory: [e_theta, e_y, x, y, psi, v]  (6 x N)
- s_ref: arc-length parameterization [m]
- kappa_ref: curvature profile [1/m]
"""

import numpy as np
import cv2 as cv
from path_planning_eugen import PathPlanning
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class TrajectoryGeneration:
    def __init__(self, ds=0.05, N_horizon=20, v_max=1.0, v_min=0.5, 
                 use_curvature_velocity=True, smooth_velocity=True):
        """
        Initialize trajectory generator.

        Args:
            ds (float): Spatial sampling distance [m].
            N_horizon (int): MPC prediction horizon length.
            v_max (float): Maximum velocity [m/s].
            v_min (float): Minimum velocity [m/s].
            use_curvature_velocity (bool): Adapt velocity based on curvature.
            smooth_velocity (bool): Apply smoothing to velocity profile.
        """
        map_img = cv.imread("data/2024_VerySmall.png")
        self.planner = PathPlanning(map_img)

        self.ds = ds
        self.N_horizon = N_horizon
        self.v_max = v_max
        self.v_min = v_min
        self.use_curvature_velocity = use_curvature_velocity
        self.smooth_velocity = smooth_velocity

        # Will be filled by generating_spatial_reference()
        self.s_ref = None
        self.x_ref = None
        self.y_ref = None
        self.psi_ref = None
        self.kappa_ref = None
        self.v_ref = None

    def time2space(self, X, Y, YAW):
        """
        Shortcut to call the underlying PathPlanning.time2space().
        """
        return self.planner.time2space(X, Y, YAW)

    # -------------------------------------------------------------
    # Velocity profile generation with smoothing
    # -------------------------------------------------------------
    def _compute_velocity_profile(self, kappa):
        """
        Compute a curvature-adaptive velocity profile with smooth transitions.

        Args:
            kappa (array): curvature profile [1/m]
        Returns:
            v (array): velocity profile [m/s]
        """
        if not self.use_curvature_velocity:
            v_profile = np.ones_like(kappa) * self.v_max
        else:

            # Smooth velocity transition: v = v_max / (1 + beta * |kappa|)
            beta = 0.8  # curvature sensitivity parameter
            v_raw = self.v_max / (1 + beta * kappa)
            v_profile = np.clip(v_raw, self.v_min, self.v_max)

        # Apply smoothing to velocity profile only
        if self.smooth_velocity and len(v_profile) > 10:
            try:
                window_length = min(25, len(v_profile) - 1)
                if window_length % 2 == 0: 
                    window_length -= 1  # Ensure odd
                if window_length >= 3:
                    v_profile = savgol_filter(v_profile, window_length, 2)
                    v_profile = np.clip(v_profile, self.v_min, self.v_max)
            except Exception as e:
                print(f"[WARNING] Velocity smoothing failed: {e}")

        return v_profile

    # -------------------------------------------------------------
    # Full trajectory generation
    # -------------------------------------------------------------
    def generating_spatial_reference(self, nodes_to_visit):
        """
        Compute full path and curvature from path planner, 
        and derive the velocity profile.

        Args:
            nodes_to_visit (list): list of graph node IDs to pass through.

        Returns:
            trajectory (6xN): [e_theta, e_y, x, y, psi, v]
            s_ref (N,): arc-length [m]
            kappa_ref (N,): curvature [1/m]
        """
        (
            s_ref,
            x_ref,
            y_ref,
            psi_ref,
            kappa_ref,
            clothoids,
        ) = self.planner.generate_path_passing_through(nodes_to_visit, step_length=self.ds)

        v_ref = self._compute_velocity_profile(kappa_ref)

        e_theta = np.zeros_like(s_ref)
        e_y = np.zeros_like(s_ref)
        #psi_ref = np.unwrap(psi_ref)
        trajectory = np.stack((e_theta, e_y, x_ref, y_ref, psi_ref, v_ref)).astype(np.float32)

        # Store internally
        self.s_ref = s_ref
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.psi_ref = psi_ref
        self.kappa_ref = kappa_ref
        self.v_ref = v_ref
        self.clothoids = clothoids

        # print(f"[INFO] Trajectory generated: {len(s_ref)} points, horizon {self.N_horizon}")
        # print(f"[INFO] Velocity smoothing: {self.smooth_velocity}")
        # print(f"[INFO] Velocity range: {v_ref.min():.3f} - {v_ref.max():.3f} m/s")
        return trajectory, s_ref, kappa_ref

    # -------------------------------------------------------------
    # Online local horizon generator (for MPC)
    # -------------------------------------------------------------
    def generating_online_spatial_ref(self, s_curr):
        """
        Generate local reference trajectory segment starting from current arc length s_curr.

        Args:
            s_curr (float): current arc length along reference [m]

        Returns:
            traj_segment (6xN_horizon): [e_theta, e_y, x, y, psi, v]
        """
        if self.s_ref is None:
            raise RuntimeError("Call generating_spatial_reference() first.")

        s_end = s_curr + (self.N_horizon - 1) * self.ds
        s_horizon = np.linspace(s_curr, s_end, self.N_horizon)
        s_horizon = np.clip(s_horizon, 0, self.s_ref[-1])

        # Interpolate all reference signals
        x_h = np.interp(s_horizon, self.s_ref, self.x_ref)
        y_h = np.interp(s_horizon, self.s_ref, self.y_ref)
        psi_h = np.interp(s_horizon, self.s_ref, self.psi_ref)
        v_h = np.interp(s_horizon, self.s_ref, self.v_ref)

        e_theta = np.zeros_like(s_horizon)
        e_y = np.zeros_like(s_horizon)

        traj_segment = np.stack((e_theta, e_y, x_h, y_h, psi_h, v_h)).astype(np.float32)
        return traj_segment

    # -------------------------------------------------------------
    # Plotting functions
    # -------------------------------------------------------------
    def plot_velocity_profile(self):
        """Plot the velocity profile along with curvature for comparison."""
        if self.s_ref is None or self.v_ref is None:
            print("[ERROR] No trajectory data available. Call generating_spatial_reference() first.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot velocity profile
        ax1.plot(self.s_ref, self.v_ref, 'b-', linewidth=2, label='Velocity')
        ax1.set_ylabel('Velocity [m/s]', fontsize=12)
        ax1.set_ylim([0.0 , 0.7])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Velocity Profile Along Path')
        
        # Plot curvature
        ax2.plot(self.s_ref, self.kappa_ref, 'r-', linewidth=2, label='Curvature')
        ax2.set_xlabel('Arc Length [m]', fontsize=12)
        ax2.set_ylabel('Curvature [1/m]', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_path_with_speed(self):
        """Plot the path colored by speed."""
        if self.x_ref is None or self.v_ref is None:
            print("[ERROR] No trajectory data available. Call generating_spatial_reference() first.")
            return

        plt.figure(figsize=(10, 8))
        
        # Create scatter plot colored by velocity
        scatter = plt.scatter(self.x_ref, self.y_ref, c=self.v_ref, 
                            cmap='viridis', s=10, alpha=0.7)
        
        plt.colorbar(scatter, label='Velocity [m/s]')
        plt.plot(self.x_ref, self.y_ref, 'k-', alpha=0.3, linewidth=1)
        plt.xlabel('X [m]', fontsize=12)
        plt.ylabel('Y [m]', fontsize=12)
        plt.title('Path Colored by Velocity')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    nodes_to_visit = [73, 97, 100, 130, 140]
    nodes_to_visit = [397, 307, 377] # round about 1 then 2nd exit

    traj_gen = TrajectoryGeneration(ds=0.1, N_horizon=100, v_max=0.35, v_min=0.3, 
                                  use_curvature_velocity=False, smooth_velocity=False)

    trajectory, s_ref, kappa_ref = traj_gen.generating_spatial_reference(nodes_to_visit)
    traj_horizon = traj_gen.generating_online_spatial_ref(s_curr=0.5)

    print("Trajectory shape:", trajectory.shape)
    print("Horizon shape:", traj_horizon.shape)

    # Draw points on path
    traj_gen.planner.draw_path_nodes(traj_gen.planner.route_list)

    # Draw whole path
    full_path = np.column_stack((traj_gen.planner.x_ref, traj_gen.planner.y_ref))
    traj_gen.planner.draw_path(full_path)

    # Draw horizon MPC path with velocity coloring
    horizon_path_xy = np.column_stack((traj_gen.x_ref, traj_gen.y_ref))
    traj_gen.planner.draw_path_gradient(horizon_path_xy, traj_gen.v_ref, thickness=5)

    # Show map
    traj_gen.planner.show_map_resized(roi_height_ratio=0.55, roi_width_ratio=0.35, scale=0.5)
    cv.waitKey(0)

    #Plot path and curvature (original)
    traj_gen.planner.plot_path_and_curvature(traj_gen.planner.x_ref, 
                                   traj_gen.planner.y_ref, 
                                   traj_gen.planner.kappa_ref, 
                                   traj_gen.planner.s_ref, 
                                   route=traj_gen.planner.route_list)

    # Plot velocity profile
    traj_gen.plot_velocity_profile()
    
    # Plot path colored by speed
    traj_gen.plot_path_with_speed()

    cv.destroyAllWindows()