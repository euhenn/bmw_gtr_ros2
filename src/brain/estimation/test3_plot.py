import numpy as np
import matplotlib.pyplot as plt
from test3 import UKFEstimator, wrap_angle


def main():

    experimentalRun = 1

    print('Loading the data file #', experimentalRun)
    experimentalData = np.genfromtxt(
        'data/mpc_{0:03d}.csv'.format(experimentalRun),
        delimiter=',',
        skip_header=1
    )

    # initial measurements
    x0   = experimentalData[0, 2]
    y0   = experimentalData[0, 3]
    yaw0 = experimentalData[0, 4]

    # initialize estimator
    ukf = UKFEstimator(x0, y0, yaw0)

    numDataPoints = experimentalData.shape[0]

    # allocate result arrays
    est_x     = np.zeros(numDataPoints)
    est_y     = np.zeros(numDataPoints)
    est_yaw   = np.zeros(numDataPoints)

    print('Running estimator...')

    for k in range(numDataPoints):

        if k == 0:
            dt = 0.0
        else:
            dt = experimentalData[k, 0] - experimentalData[k-1, 0]

        # measurements
        meas_x   = experimentalData[k, 2]
        meas_y   = experimentalData[k, 3]
        meas_yaw = experimentalData[k, 4]

        # inputs
        speed = experimentalData[k, 5]
        gamma = experimentalData[k, 6]

        x, y, yaw = ukf.step(dt, speed, gamma, meas_x, meas_y, meas_yaw)

        est_x[k]   = x
        est_y[k]   = y
        est_yaw[k] = yaw

    print('Done.')

    # ----------------------------------------------------------------------
    # WRAP ANGLE
    # ----------------------------------------------------------------------
    est_yaw = wrap_angle(est_yaw)

    # ground truth / measurements from log
    t_vec = experimentalData[:, 0]
    true_x   = experimentalData[:, 2]
    true_y   = experimentalData[:, 3]
    true_yaw = experimentalData[:, 4]

    # ----------------------------------------------------------------------
    # Errors
    # ----------------------------------------------------------------------
    err_x = est_x - true_x
    err_y = est_y - true_y
    err_yaw = wrap_angle(est_yaw - true_yaw)

    print("Final error:")
    print("   x error =", err_x[-1])
    print("   y error =", err_y[-1])
    print("   yaw error =", err_yaw[-1])

    avg_x   = np.mean(np.abs(err_x))
    avg_y   = np.mean(np.abs(err_y))
    avg_yaw = np.mean(np.abs(err_yaw))

    print("Average:")
    print("   x =", avg_x)
    print("   y =", avg_y)
    print("   yaw =", avg_yaw)
    print("Score =", avg_x + avg_y + avg_yaw)


    # ==========================================================================
    # PLOTS
    # ==========================================================================
    print("Generating plots...")

    # ---- TOP VIEW ----
    fig1, ax1 = plt.subplots()
    ax1.plot(true_x, true_y, 'k.', markersize=2, label='GPS meas')
    ax1.plot(est_x, est_y, 'b-', linewidth=1.3, label='UKF estimate')
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend()
    ax1.set_title("Top View: Position Tracking")

    # ---- TIME SERIES ----
    fig2, ax = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

    ax[0].plot(t_vec, true_x, 'k:', label="x meas")
    ax[0].plot(t_vec, est_x, 'b-', label="x est")
    ax[0].legend()
    ax[0].set_ylabel("x [m]")

    ax[1].plot(t_vec, true_y, 'k:', label="y meas")
    ax[1].plot(t_vec, est_y, 'b-', label="y est")
    ax[1].set_ylabel("y [m]")

    ax[2].plot(t_vec, true_yaw, 'k:', label="yaw meas")
    ax[2].plot(t_vec, est_yaw, 'b-', label="yaw est")
    ax[2].set_ylabel("yaw [rad]")

    ax[3].plot(t_vec, experimentalData[:, 6], 'g-', label="steering")
    ax[3].set_ylabel("steering [rad]")

    ax[4].plot(t_vec, experimentalData[:, 5], 'g-', label="speed")
    ax[4].set_ylabel("speed [m/s]")
    ax[4].set_xlabel("Time [s]")

    # ---- ERROR FIGURES ----
    fig3, axE = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axE[0].plot(t_vec, err_x, 'r-')
    axE[0].axhline(0, color='k', linewidth=0.5)
    axE[0].set_ylabel("err x [m]")

    axE[1].plot(t_vec, err_y, 'r-')
    axE[1].axhline(0, color='k', linewidth=0.5)
    axE[1].set_ylabel("err y [m]")

    axE[2].plot(t_vec, err_yaw, 'r-')
    axE[2].axhline(0, color='k', linewidth=0.5)
    axE[2].set_ylabel("err yaw [rad]")
    axE[2].set_xlabel("Time [s]")

    print("Showing plots...")
    plt.show()


if __name__ == "__main__":
    main()
