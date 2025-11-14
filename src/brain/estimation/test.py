import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# *** NEW IMPORTS ***
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


# -------------------------------------------------------------------------
#  Define the bicycle model for FilterPy UKF
# -------------------------------------------------------------------------
B = 0.267  # wheelbase


def fx(state, dt, u):
    """Process model.
    state = [x, y, theta]
    u = [speed, steeringAngle]
    """
    x, y, theta = state
    v, gamma = u

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(gamma) * dt / B

    return np.array([x_new, y_new, theta_new])


def hx(state):
    """Measurement model: sensor gives x, y only."""
    x, y, theta = state
    return np.array([x, y])


# -------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------
def estInitialize(first_meas_x, first_meas_y, first_meas_yaw):
    # UKF state = [x, y, theta]
    x0 = np.array([first_meas_x, first_meas_y, first_meas_yaw])

    # Merwe sigma points (standard choices)
    points = MerweScaledSigmaPoints(n=3, alpha=0.3, beta=2.0, kappa=1.0)

    ukf = UnscentedKalmanFilter(
        dim_x=3,
        dim_z=2,
        fx=fx,
        hx=hx,
        dt=0.01,  # overridden manually in estRun
        points=points
    )

    # Initial state
    ukf.x = x0

    # Initial covariance
    ukf.P = np.diag([0.05**2, 0.05**2, (5*np.pi/180)**2])

    # Process noise (same order as your original V)
    ukf.Q = np.diag([0.004, 0.004, 0.001])

    # Measurement noise (same W)
    ukf.R = np.diag([0.05**2, 0.05**2])

    return ukf


# -------------------------------------------------------------------------
# UKF step
# -------------------------------------------------------------------------
def estRun(time, dt, internalStateIn, steeringAngle, speed, measurement):
    ukf = internalStateIn

    # --- Handle missing measurements ---
    missing_meas = (np.isnan(measurement[0]) or np.isnan(measurement[1]))

    # UKF input = [speed, steeringAngle]
    control = np.array([speed, steeringAngle])

    # --- Prediction step ---
    ukf.predict(dt=dt, u=control)

    # --- Update if measurement available ---
    if not missing_meas:
        z = np.array([measurement[0], measurement[1]])
        ukf.update(z)

    # Extract state
    x = ukf.x[0]
    y = ukf.x[1]
    theta = ukf.x[2]

    return x, y, theta, ukf


# -------------------------------------------------------------------------
# MAIN (same as before, only internalState becomes UKF object)
# -------------------------------------------------------------------------
def main():

    experimentalRun = 1
    print("Loading data", experimentalRun)
    experimentalData = np.genfromtxt('data/mpc_{0:03d}.csv'.format(experimentalRun),
                                     delimiter=',', skip_header=1)

    print("Initializing UKF...")
    internalState = estInitialize(
        experimentalData[0, 2],
        experimentalData[0, 3],
        experimentalData[0, 4]
    )

    numDataPoints = experimentalData.shape[0]

    estimatedPosition_x = np.zeros(numDataPoints)
    estimatedPosition_y = np.zeros(numDataPoints)
    estimatedAngle      = np.zeros(numDataPoints)

    print("Running UKF...")

    for k in range(numDataPoints):
        if k == 0:
            dt = 0.0
        else:
            dt = experimentalData[k, 0] - experimentalData[k-1, 0]

        t       = experimentalData[k, 0]
        measx   = experimentalData[k, 2]
        measy   = experimentalData[k, 3]
        measyaw = experimentalData[k, 4]
        gamma   = experimentalData[k, 6]
        speed   = experimentalData[k, 5]

        x, y, theta, internalState = estRun(
            t, dt, internalState,
            gamma, speed,
            (measx, measy)
        )

        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k] = theta

    print("Done.")

    # Wrap angle
    estimatedAngle = np.mod(estimatedAngle + np.pi, 2*np.pi) - np.pi

    posErr_x = estimatedPosition_x - experimentalData[:, 2]
    posErr_y = estimatedPosition_y - experimentalData[:, 3]
    angErr   = np.mod(estimatedAngle - experimentalData[:, 4] + np.pi, 2*np.pi) - np.pi

    print("Final error:")
    print("   pos x =", posErr_x[-1])
    print("   pos y =", posErr_y[-1])
    print("   angle =", angErr[-1])

    # Average score
    ax = np.mean(np.abs(posErr_x))
    ay = np.mean(np.abs(posErr_y))
    ath = np.mean(np.abs(angErr))
    print("Average score =", ax + ay + ath)

    # Plotting identical to your version
    print("Generating plots")
    figTop, axTop = plt.subplots()
    axTop.plot(estimatedPosition_x, estimatedPosition_y, 'b', label="est")
    axTop.plot(experimentalData[:,2], experimentalData[:,3], 'k:', label="meas")
    axTop.legend()
    axTop.set_xlabel("x [m]")
    axTop.set_ylabel("y [m]")

    plt.show()


if __name__ == "__main__":
    main()
