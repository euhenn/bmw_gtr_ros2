
"""
CODE FOR GENERETING ELLIPSE AND SCURVE REFERNCE

provides wih x,y position, velocity and the sum of heading and slipping angle (actual angle we have)
             time needed for executing desired trajectory wrt sampling time of the controller

we provide  omega - angle that we use to compute the reference (and its time space)
            dt - desired sampling time of controller
            N_horizon - number of horizon steps 
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from casadi import sin, cos, tan, arctan
import cv2 as cv
from path_planning_eugen import PathPlanning 

class TrajectoryGeneration:
    def __init__(self, ds, N_horizon):
        track = cv.imread('data/final_map.png')
        self.planner = PathPlanning(track)
        self.N_horizon = N_horizon
        self.ds = ds

        self.cs_x = None
        self.cs_y = None
        self.S_length = None

    def generating_dense_spatial_ref(self):
        self.planner.generate_path_passing_through(self.nodes_to_visit, step_length=0.01)
        path_dense = np.array(self.planner.path)  
        Xd = path_dense[:, 0]
        Yd = path_dense[:, 1]

        # 2) cumulative arc-length along the dense path
        dxy = np.diff(path_dense, axis=0)
        segment_length = np.hypot(dxy[:, 0], dxy[:, 1])
        s_dense = np.concatenate(([0.0], np.cumsum(segment_length)))  # len = M_dense

        # Make sure last s matches total length
        S_length = s_dense[-1]
        if S_length <= 0:
            raise RuntimeError("Zero length path")

        # 3) build splines X(s), Y(s)
        self.cs_x = CubicSpline(s_dense, Xd)
        self.cs_y = CubicSpline(s_dense, Yd)
        self.S_length = S_length 

        return self.cs_x, self.cs_y, self.S_length

    def generating_spatial_reference(self, nodes_to_visit):
        # Generate the path
        self.nodes_to_visit = nodes_to_visit
        cs_x, cs_y, S_length = self.generating_dense_spatial_ref()
        # 4) uniform arc-length grid
        s_uniform = np.arange(0.0, S_length, self.ds)
        if s_uniform[-1] < S_length:
            s_uniform = np.concatenate((s_uniform, [S_length]))  # include final point

        # 5) evaluate spline at uniform s
        Xu = cs_x(s_uniform)
        Yu = cs_y(s_uniform)
        e_theta = np.zeros(Xu.shape)
        e_y= np.zeros(Xu.shape)

        # 6) compute derivatives and heading and curvature
        dx_ds = cs_x.derivative(1)(s_uniform)
        dy_ds = cs_y.derivative(1)(s_uniform)
        ddx_ds2 = cs_x.derivative(2)(s_uniform)
        ddy_ds2 = cs_y.derivative(2)(s_uniform)

        theta = np.arctan2(dy_ds, dx_ds)   # heading
        theta = np.unwrap(theta)
        self.theta = theta
        denom = (np.hypot(dx_ds, dy_ds)**3) 
        # avoid division by zero
        denom = np.maximum(denom, 1e-9)
        kappa = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom
        v = 0.5*np.ones(Xu.shape)

        trajectory = np.stack((e_theta, e_y, Xu, Yu, theta, v)).astype(np.float32)  # (M,3)

        return trajectory, s_uniform, kappa

    
    def generating_online_spatial_ref(self, s):
        s_horizon = np.linspace(s, s+ self.N_horizon*self.ds -self.ds , self.N_horizon)
        s_horizon = np.clip(s_horizon, 0, self.S_length)
        
        ds = s_horizon[2] - s_horizon[1]
        #print(ds)
        # 5) evaluate spline at uniform s
        Xu = self.cs_x(s_horizon)
        Yu = self.cs_y(s_horizon)
        e_theta = np.zeros(Xu.shape)
        e_y= np.zeros(Xu.shape)

        # 6) compute derivatives and heading and curvature
        dx_ds = self.cs_x.derivative(1)(s_horizon)
        dy_ds = self.cs_y.derivative(1)(s_horizon)
        ddx_ds2 = self.cs_x.derivative(2)(s_horizon)
        ddy_ds2 = self.cs_y.derivative(2)(s_horizon)

        theta = np.arctan2(dy_ds, dx_ds)   # heading
        theta = np.unwrap(theta)
        
        denom = (np.hypot(dx_ds, dy_ds)**3) 
        # avoid division by zero
        denom = np.maximum(denom, 1e-9)
        kappa_horizon = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom
        v = 1*np.ones(Xu.shape)

        trajectory = np.stack((e_theta, e_y, Xu, Yu, theta, v)).astype(np.float32) 
        print(trajectory.shape) # (M,3)

        return trajectory
        

    def generating_time_reference(self, N_horizon, ds, nodes_to_visit): 
        self.planner.generate_path_passing_through(nodes_to_visit, step_length=0.001)
        path = []
        yaw_path = []
        for i in range(len(self.planner.path)-1):
            x, y = self.planner.path[i]
            x_next, y_next = self.planner.path[i+1]
            yaw = np.arctan2(y_next - y, x_next - x)
            yaw_path.append((yaw))
        yaw_path = np.unwrap(yaw_path)
        for (x, y), yaw in zip(self.planner.path[:-1], yaw_path):
            path.append((x, y, yaw))
        # Add last point with previous yaw
        path.append((*self.planner.path[-1], yaw_path[-1]))
        trajectory = np.array(path, dtype=np.float32)


        N = trajectory.shape[0] - N_horizon
        x = np.zeros(N+N_horizon)
        y= np.zeros(N+N_horizon)
        theta = np.zeros(N+N_horizon)
        for i in range(trajectory.shape[0] ):
            x[i], y[i], theta[i] = trajectory[i]
            theta[i] = theta[i]
            #print(f"Row {i}: x={x}, y={y}, theta={theta}")
        #for j in range(N_horizon):
            #x[N+j], y[N+j], theta[N+j] = trajectory[N-1]
            #print(f"Row {i}: x={x}, y={y}, theta={theta}")
        
        y_ref = np.vstack((x, y, theta)).T
        print(y_ref.shape)
        return y_ref, N



def plot_trajectory_in_space(y_ref, label='Trajectory', title='Reference Trajectory'):
    plt.figure()
    plt.plot(y_ref[2, :], y_ref[3,:], label=label)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_full_trajectory(yref, ds, label='Trajectory', title='Reference Trajectory'):
    ny= yref.shape [0]
    N= yref.shape[1]
    timestamps = np.arange(N) * ds
    fig, ax = plt.subplots(ny, 1, sharex=True, figsize=(6, 8))
    fig.suptitle('States and Control over Time', fontsize=14, y=0.97)
    labels = ["x", "y", "heading+slipping angle"]
    for i in range(ny):
        ax[i].plot(timestamps, yref[ i, :], '--', label='Reference')
        #ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("arc of length [m]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    N_horizon = 50
    ds = 0.06
    
    print("Initializing Trajectory Generator...")
    track = TrajectoryGeneration(ds, N_horizon)
    
    # Single path test
    nodes = [73, 97, 125]
    print(f"\n{'='*50}")
    print(f"Generating trajectory with nodes: {nodes}")
    print(f"{'='*50}")
    
    # Generate trajectory
    trajectory, s_uniform, kappa = track.generating_spatial_reference(nodes)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Path length: {s_uniform[-1]:.2f} meters")
    print(f"Number of points: {len(s_uniform)}")
    print(f"Sampling distance: {ds} meters")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Trajectory Generation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spatial trajectory (main plot)
    axes[0,0].plot(trajectory[2, :], trajectory[3, :], 'b-', linewidth=2, label='Reference path')
    axes[0,0].plot(trajectory[2, 0], trajectory[3, 0], 'go', markersize=10, label='Start', markeredgecolor='black')
    axes[0,0].plot(trajectory[2, -1], trajectory[3, -1], 'ro', markersize=10, label='End', markeredgecolor='black')
    
    # Mark node positions if available
    for i, node in enumerate(nodes):
        axes[0,0].plot(trajectory[2, i*len(s_uniform)//len(nodes)], 
                      trajectory[3, i*len(s_uniform)//len(nodes)], 
                      's', color='orange', markersize=8, label=f'Node {node}' if i == 0 else "")
    
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
    
    # 5. Errors (should be zero for reference)
    axes[1,1].plot(s_uniform, trajectory[0, :], 'orange', linewidth=2, label='Heading error')
    axes[1,1].plot(s_uniform, trajectory[1, :], 'brown', linewidth=2, label='Cross-track error')
    axes[1,1].set_xlabel('Arc Length [m]')
    axes[1,1].set_ylabel('Error [rad/m]')
    axes[1,1].set_title('Reference Errors')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Online reference generation demonstration
    s_test_points = [0.5, s_uniform[-1]/3, 2*s_uniform[-1]/3]
    colors = ['red', 'blue', 'green']
    
    axes[1,2].plot(trajectory[2, :], trajectory[3, :], 'k-', alpha=0.3, label='Full trajectory')
    
    for i, s_current in enumerate(s_test_points):
        if s_current < s_uniform[-1]:
            online_traj = track.generating_online_spatial_ref(s_current)
            axes[1,2].plot(online_traj[2, :], online_traj[3, :], 
                          color=colors[i], linewidth=3, 
                          label=f'Horizon @ s={s_current:.1f}m')
            axes[1,2].plot(online_traj[2, 0], online_traj[3, 0], 
                          'o', color=colors[i], markersize=8, markeredgecolor='black')
    
    axes[1,2].set_xlabel('X [m]')
    axes[1,2].set_ylabel('Y [m]')
    axes[1,2].set_title('Online Reference Generation\n(Prediction Horizon)')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nPath Statistics:")
    print(f"  - Total length: {s_uniform[-1]:.2f} m")
    print(f"  - Max curvature: {np.max(np.abs(kappa)):.3f} 1/m")
    print(f"  - Min curvature: {np.min(np.abs(kappa)):.3f} 1/m")
    print(f"  - Max heading: {np.degrees(np.max(trajectory[4, :])):.1f}°")
    print(f"  - Min heading: {np.degrees(np.min(trajectory[4, :])):.1f}°")
    print(f"  - Velocity: constant {trajectory[5, 0]:.1f} m/s")
    
    # Test online reference generation
    print(f"\n{'='*50}")
    print("Testing Online Reference Generation")
    print(f"{'='*50}")
    
    test_s = 1.5  # Test point at 1.5 meters along the path
    if test_s <= s_uniform[-1]:
        online_ref = track.generating_online_spatial_ref(test_s)
        print(f"Online reference at s={test_s:.1f}m:")
        print(f"  - Returns {online_ref.shape[1]} horizon points")
        print(f"  - Start position: ({online_ref[2, 0]:.2f}, {online_ref[3, 0]:.2f})")
        print(f"  - End position: ({online_ref[2, -1]:.2f}, {online_ref[3, -1]:.2f})")
        print(f"  - Start heading: {np.degrees(online_ref[4, 0]):.1f}°")
    else:
        print(f"Test point s={test_s:.1f}m exceeds path length!")
    
    # Also show the original simple plots
    print(f"\n{'='*50}")
    print("Generating Basic Plots")
    print(f"{'='*50}")
    
    #plot_trajectory_in_space(trajectory, label='Generated Path', title='Spatial Trajectory')
    #plot_full_trajectory(trajectory, ds, label='State Evolution', title='Trajectory States')
