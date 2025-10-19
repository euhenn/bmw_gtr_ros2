import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from casadi import sin, cos, tan, arctan
import cv2 as cv
from path_planning1 import PathPlanning 

class TrajectoryGeneration:
    def __init__(self):
        track = cv.imread('data/final_map.png')
        self.planner = PathPlanning(track)

    def generating_reference(self, nodes_to_visit):
        # Generate the path
        ds = 0.02
        self.planner.generate_path_passing_through(nodes_to_visit, step_length=ds)
        path_with_yaw = []
        for i in range(len(self.planner.path)-1):
            x, y = self.planner.path[i]
            x_next, y_next = self.planner.path[i+1]
            yaw = np.rad2deg(np.arctan2(y_next - y, x_next - x))
            path_with_yaw.append((x, y, yaw))

        # Add last point with previous yaw
        path_with_yaw.append((*self.planner.path[-1], yaw))
        trajectory = np.array(path_with_yaw, dtype=np.float32)

        # Draw the path
        #self.planner.draw_path()

        # Scale down the image before displaying
        #scale = 0.25  # 50% of original size
        #resized_map = cv.resize(self.planner.map, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

        # Show the scaled image
        #cv.imshow("Generated Path", resized_map)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        return trajectory

        
    def time_reference(self, N_horizon, nodes_to_visit): 
        trajectory = self.generating_reference(nodes_to_visit)
        N = trajectory.shape[0] 
        x = np.zeros(N+N_horizon)
        y= np.zeros(N+N_horizon)
        theta = np.zeros(N+N_horizon)
        for i in range(N):
            x[i], y[i], theta[i] = trajectory[i]
            theta[i] = np.deg2rad(theta[i])
            #print(f"Row {i}: x={x}, y={y}, theta={theta}")
        for j in range(N_horizon):
            x[N+j], y[N+j], theta[N+j] = trajectory[N-1]
            #print(f"Row {i}: x={x}, y={y}, theta={theta}")
        
        y_ref = np.stack((x, y, theta))
        print(y_ref.shape)
        return y_ref, N

    def spatial_reference(self, N_horizon): 
        N = self.trajectory.shape[0]
        x = np.zeros(N+N_horizon)
        y= np.zeros(N+N_horizon)
        theta = np.zeros(N+N_horizon)
        for i in range(N):
            x[i], y[i], theta[i] = self.trajectory[i]
            theta[i] = np.deg2rad(theta[i])
            #print(f"Row {i}: x={x}, y={y}, theta={theta}")
        for j in range(N_horizon):
            x[N+j], y[N+j], theta[N+j] = self.trajectory[N-1]
            #print(f"Row {i}: x={x}, y={y}, theta={theta}")
        
        y_ref = np.stack((x, y, theta))
        print(y_ref.shape)
        return y_ref, N


def plot_trajectory_in_space(y_ref, label='Trajectory', title='Reference Trajectory'):
    plt.figure()
    plt.plot(y_ref[0, :], y_ref[1, :], label=label)
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
        ax[i].plot(timestamps, yref[i, :], '--', label='Reference')
        #ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("arc of length [m]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    track = TrajectoryGeneration()
    N_horizon = 10
    nodes = [73,91]
    y_ref, N = track.time_reference(N_horizon, nodes)
    plot_trajectory_in_space(y_ref, label='Car Path')
    plot_full_trajectory(y_ref, 0.1)
