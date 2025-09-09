import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from casadi import sin, cos, tan, arctan

"""
CODE FOR GENERETING ELLIPSE AND SCURVE REFERNCE

provides wih x,y position, velocity and the sum of heading and slipping angle (actual angle we have)
             time needed for executing desired trajectory wrt sampling time of the controller

we provide  omega - angle that we use to compute the reference (and its time space)
            dt - desired sampling time of controller
            N_horizon - number of horizon steps 
"""
class TimeTraj:
    def __init__(self):
        self.trajectory =np.load("trajectory.npy")
        
    def full_reference(self, N_horizon): 
        N = self.trajectory.shape[0]
        x = np.zeros(N+N_horizon)
        y= np.zeros(N+N_horizon)
        theta = np.zeros(N+N_horizon)
        for i in range(N):
            x[i], y[i], theta[i] = self.trajectory[i]
            theta[i] = np.deg2rad(theta[i])
            print(f"Row {i}: x={x}, y={y}, theta={theta}")
        for j in range(N_horizon):
            x[N+j], y[N+j], theta[N+j] = self.trajectory[N-1]
            print(f"Row {i}: x={x}, y={y}, theta={theta}")
        
        y_ref = np.stack((x, y, theta))
        print(y_ref.shape)
        return y_ref, N

class SpatialTraj:
    def __init__(self):
        self.trajectory =np.load("trajectory.npy")
        
    def full_reference(self, N_horizon): 
        N = self.trajectory.shape[0]
        x = np.zeros(N+N_horizon)
        y= np.zeros(N+N_horizon)
        theta = np.zeros(N+N_horizon)
        for i in range(N):
            x[i], y[i], theta[i] = self.trajectory[i]
            theta[i] = np.deg2rad(theta[i])
            print(f"Row {i}: x={x}, y={y}, theta={theta}")
        for j in range(N_horizon):
            x[N+j], y[N+j], theta[N+j] = self.trajectory[N-1]
            print(f"Row {i}: x={x}, y={y}, theta={theta}")
        
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


if __name__ == "__main__":
    sim = Simulation()
    N_horizon = 10
    y_ref, N = sim.full_reference(N_horizon)
    plot_trajectory_in_space(y_ref, label='Car Path')
