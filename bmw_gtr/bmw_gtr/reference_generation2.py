
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
from path_planning1 import PathPlanning 

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
    
    N_horizon = 50
    ds = 0.06
    track = TrajectoryGeneration(ds, N_horizon)
    nodes = [73,91,125,141]
    nodes = [410,391,352,377,418]
    nodes = [125,160,150,133,126]
    nodes = [73,91,125]
    trajectory, s_uniform, kappa = track.generating_spatial_reference(nodes)

    yaw_hist = np.load('yaw_hist.npy')
    n = yaw_hist.shape[0]
    plt.plot(yaw_hist, label='actual')
    plt.plot(trajectory[4,:n], label='ref')
    plt.legend()
    plt.show()
    exit(0)
    
    s_current = 2.0
    #traj_local, s_local, kappa_local = track.generating_online_spatial_ref(s_current)
    #np.save("path1.npy", y_ref) #np.load("path.npy") 
    plot_full_trajectory(trajectory, ds, label='Car Path')
    plot_trajectory_in_space(trajectory, label='Car Path')
