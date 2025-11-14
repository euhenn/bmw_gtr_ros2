import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


# ---------- helper functions ----------
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# ---------- bicycle model ----------
B = 0.267   # wheelbase [m]


def fx(state, dt, u):
    """Process model: state = [x, y, theta], u = [v, gamma]."""
    x, y, theta = state
    v, gamma = u

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(gamma) * dt / B
    theta_new = wrap_angle(theta_new)

    return np.array([x_new, y_new, theta_new])


def hx(state):
    """Measurement model: z = [x, y, yaw]."""
    x, y, theta = state
    return np.array([x, y, theta])


# ---------- UKF angle-aware mean & residuals ----------
def state_mean(sigmas, Wm):
    """Mean of state sigma points with angle in index 2."""
    x = np.zeros(3)
    x[0] = np.dot(Wm, sigmas[:, 0])
    x[1] = np.dot(Wm, sigmas[:, 1])

    # angle mean using sin/cos
    sin_sum = np.dot(Wm, np.sin(sigmas[:, 2]))
    cos_sum = np.dot(Wm, np.cos(sigmas[:, 2]))
    x[2] = np.arctan2(sin_sum, cos_sum)
    return x


def meas_mean(sigmas, Wm):
    """Mean of measurement sigma points with angle in index 2."""
    z = np.zeros(3)
    z[0] = np.dot(Wm, sigmas[:, 0])
    z[1] = np.dot(Wm, sigmas[:, 1])

    sin_sum = np.dot(Wm, np.sin(sigmas[:, 2]))
    cos_sum = np.dot(Wm, np.cos(sigmas[:, 2]))
    z[2] = np.arctan2(sin_sum, cos_sum)
    return z


def residual_x(a, b):
    """State residual a - b with angle wrapping."""
    y = a - b
    y[2] = wrap_angle(y[2])
    return y


def residual_z(a, b):
    """Measurement residual a - b with angle wrapping."""
    y = a - b
    y[2] = wrap_angle(y[2])
    return y


# ---------- initialization ----------
def estInitialize(first_meas_x, first_meas_y, first_meas_yaw):
    x0 = np.array([first_meas_x, first_meas_y, first_meas_yaw])

    # Sigma points
    points = MerweScaledSigmaPoints(
        n=3,
        alpha=0.0001,
        beta=2.0,
        kappa=0.0
    )

    ukf = UnscentedKalmanFilter(
        dim_x=3,
        dim_z=3,
        dt=0.01,         # overridden each call in estRun
        hx=hx,
        fx=fx,
        points=points,
        x_mean_fn=state_mean,
        z_mean_fn=meas_mean,
        residual_x=residual_x,
        residual_z=residual_z
    )

    # Initial state and covariance
    ukf.x = x0
    ukf.P = np.diag([
        0.05**2,                 # x
        0.05**2,                 # y
        (5*np.pi/180.0)**2       # yaw
    ])

    # Process noise Q (on [x,y,yaw], representing unmodeled accel/steer noise)
    ukf.Q = np.diag([
        0.01**2,                 # x process noise
        0.01**2,                 # y process noise
        (2*np.pi/180.0)**2       # yaw process noise
    ])

    # Measurement noise R (GPS + yaw)
    sigma_x_meas   = 1e-3
    sigma_y_meas   = 1e-3
    sigma_yaw_meas = 1*np.pi/180.0
    ukf.R = np.diag([
        sigma_x_meas**2,
        sigma_y_meas**2,
        sigma_yaw_meas**2
    ])

    return ukf


# ---------- UKF wrapper ----------
def estRun(time, dt, internalStateIn, steeringAngle, speed, measurement_xy, measurement_yaw):
    """
    state: [x, y, theta]
    inputs: steeringAngle (gamma), speed (v)
    measurements: x, y, yaw
    """
    ukf = internalStateIn

    # control input
    u = np.array([speed, steeringAngle])

    # prediction
    ukf.predict(dt=dt, u=u)

    # check for missing measurements
    if (np.isnan(measurement_xy[0]) or
        np.isnan(measurement_xy[1]) or
        np.isnan(measurement_yaw)):
        # no update, just prediction
        x = ukf.x[0]
        y = ukf.x[1]
        theta = ukf.x[2]
        return x, y, theta, ukf

    # update with GPS + yaw
    z = np.array([measurement_xy[0], measurement_xy[1], measurement_yaw])
    ukf.update(z)

    x = ukf.x[0]
    y = ukf.x[1]
    theta = ukf.x[2]

    return x, y, theta, ukf


# ---------- main ----------
def main():

    experimentalRun = 1

    print('Loading the data file #', experimentalRun)
    experimentalData = np.genfromtxt(
        'data/mpc_{0:03d}.csv'.format(experimentalRun),
        delimiter=',',
        skip_header=1
    )

    print('Running the initialization')
    internalState = estInitialize(
        experimentalData[0, 2],   # x0
        experimentalData[0, 3],   # y0
        experimentalData[0, 4]    # yaw0
    )

    numDataPoints = experimentalData.shape[0]

    estimatedPosition_x = np.zeros(numDataPoints)
    estimatedPosition_y = np.zeros(numDataPoints)
    estimatedAngle      = np.zeros(numDataPoints)

    print('Running the system')

    for k in range(numDataPoints):

        if k == 0:
            dt = 0.0
        else:
            dt = experimentalData[k, 0] - experimentalData[k-1, 0]

        t       = experimentalData[k, 0]
        measx   = experimentalData[k, 2]
        measy   = experimentalData[k, 3]
        measyaw = experimentalData[k, 4]
        speed   = experimentalData[k, 5]
        gamma   = experimentalData[k, 6]

        x, y, theta, internalState = estRun(
            t, dt, internalState, gamma, speed, (measx, measy), measyaw
        )

        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k]      = theta

    print('Done running')

    # wrap estimated angle
    estimatedAngle = wrap_angle(estimatedAngle)

    t_vec   = experimentalData[:, 0]
    true_x  = experimentalData[:, 2]
    true_y  = experimentalData[:, 3]
    true_yaw = experimentalData[:, 4]

    posErr_x = estimatedPosition_x - true_x
    posErr_y = estimatedPosition_y - true_y
    angErr   = wrap_angle(estimatedAngle - true_yaw)

    print('Final error: ')
    print('   pos x =', posErr_x[-1], 'm')
    print('   pos y =', posErr_y[-1], 'm')
    print('   angle =', angErr[-1], 'rad')

    ax  = np.mean(np.abs(posErr_x))
    ay  = np.mean(np.abs(posErr_y))
    ath = np.mean(np.abs(angErr))
    score = ax + ay + ath

    print('average error:')
    print('   pos x =', ax,  'm')
    print('   pos y =', ay,  'm')
    print('   angle =', ath, 'rad')
    print('average score:', score)

    # ---------- Plots ----------
    print('Generating plots')

    figTopView, axTopView = plt.subplots(1, 1)
    axTopView.plot(true_x, true_y, 'k:.', label='meas/true')
    axTopView.plot(estimatedPosition_x, estimatedPosition_y, 'b-', label='UKF est')
    axTopView.legend()
    axTopView.set_xlabel('x-position [m]')
    axTopView.set_ylabel('y-position [m]')

    figHist, axHist = plt.subplots(5, 1, sharex=True)

    axHist[0].plot(t_vec, true_x, 'k:', label='meas x')
    axHist[0].plot(t_vec, estimatedPosition_x, 'b-', label='est x')
    axHist[0].legend()

    axHist[1].plot(t_vec, true_y, 'k:', label='meas y')
    axHist[1].plot(t_vec, estimatedPosition_y, 'b-', label='est y')

    axHist[2].plot(t_vec, true_yaw, 'k:', label='meas yaw')
    axHist[2].plot(t_vec, estimatedAngle, 'b-', label='est yaw')

    axHist[3].plot(t_vec, experimentalData[:, 6], 'g-', label='steering')
    axHist[4].plot(t_vec, experimentalData[:, 5], 'g-', label='speed')

    axHist[-1].set_xlabel('Time [s]')
    axHist[0].set_ylabel('x [m]')
    axHist[1].set_ylabel('y [m]')
    axHist[2].set_ylabel('yaw [rad]')
    axHist[3].set_ylabel('δ [rad]')
    axHist[4].set_ylabel('v [m/s]')

    # ---------- Error Plots ----------
    figError, axErr = plt.subplots(3, 1, sharex=True)

    axErr[0].plot(t_vec, posErr_x, 'r-', label='x error')
    axErr[0].axhline(0, color='k', linewidth=0.5)
    axErr[0].set_ylabel('err x [m]')
    axErr[0].legend()

    axErr[1].plot(t_vec, posErr_y, 'r-', label='y error')
    axErr[1].axhline(0, color='k', linewidth=0.5)
    axErr[1].set_ylabel('err y [m]')
    axErr[1].legend()

    axErr[2].plot(t_vec, angErr, 'r-', label='yaw error')
    axErr[2].axhline(0, color='k', linewidth=0.5)
    axErr[2].set_ylabel('err yaw [rad]')
    axErr[2].set_xlabel('Time [s]')
    axErr[2].legend()

    print('Done')
    plt.show()


if __name__ == "__main__":
    main()
