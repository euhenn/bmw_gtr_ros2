import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
# Hard mode: NO OTHER IMPORTS


# ---------- helper functions ----------
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# ---------- initialization ----------
def estInitialize(first_meas_x, first_meas_y, first_meas_yaw):
    x = first_meas_x
    y = first_meas_y
    theta = first_meas_yaw

    # initial covariance (quite confident)
    Pm = np.diag([
        0.05**2,                 # x
        0.05**2,                 # y
        (5*np.pi/180.0)**2       # yaw
    ])
    return [x, y, theta, Pm]


# ---------- UKF ----------
def estRun(time, dt, internalStateIn, steeringAngle, speed, measurement_xy, measurement_yaw):
    # state: [x, y, theta]
    # inputs: steeringAngle (gamma), speed (v)
    # measurements: x, y, yaw

    B = 0.267   # wheelbase [m]

    # Process noise (on speed and steering)
    sigma_v_proc   = 0.03               # m/s
    sigma_gam_proc = 2*np.pi/180.0      # rad
    V = np.diag([sigma_v_proc**2, sigma_gam_proc**2])

    # Measurement noise (on x, y, yaw)
    sigma_x_meas   = 1e-3
    sigma_y_meas   = 1e-3
    sigma_yaw_meas = 1*np.pi/180
    W = np.diag([sigma_x_meas**2, sigma_y_meas**2, sigma_yaw_meas**2])

    # Means of noise
    V_mean = np.zeros((2,1))
    W_mean = np.zeros((3,1))

    # Previous state & covariance
    x_last = internalStateIn[0]
    y_last = internalStateIn[1]
    th_last = internalStateIn[2]
    Pm = internalStateIn[3]

    last_state = np.array([[x_last],
                           [y_last],
                           [th_last]])

    # Augmented state: [x, y, theta, v_noise, gamma_noise, meas_noise_x, meas_noise_y, meas_noise_yaw]
    tmp_state = np.concatenate((last_state, V_mean, W_mean), axis=0)  # (8,1)
    tmp_var = scipy.linalg.block_diag(Pm, V, W)  # (8x8)

    n_aug = tmp_state.shape[0]   # 8
    num_sigma = 2 * n_aug        # 16

    # Sigma points for augmented state
    S = scipy.linalg.sqrtm(n_aug * tmp_var)   # (8x8)
    s_new = np.zeros((num_sigma, n_aug, 1))   # (16,8,1)

    for i in range(n_aug):
        s_new[i]        = tmp_state + S[:, i, np.newaxis]
        s_new[i + n_aug] = tmp_state - S[:, i, np.newaxis]

    # ----- Prediction step -----
    s_xp = np.zeros((num_sigma, 3, 1))  # predicted state sigmas: [x, y, theta]

    for i in range(num_sigma):
        x_i   = s_new[i, 0, 0]
        y_i   = s_new[i, 1, 0]
        th_i  = s_new[i, 2, 0]
        v_n   = s_new[i, 3, 0]
        g_n   = s_new[i, 4, 0]

        v_i      = speed + v_n
        gamma_i  = steeringAngle + g_n

        # simple kinematic bicycle
        x_pred  = x_i + v_i * np.cos(th_i) * dt
        y_pred  = y_i + v_i * np.sin(th_i) * dt
        th_pred = th_i + v_i * np.tan(gamma_i) * dt / B
        th_pred = wrap_angle(th_pred)

        s_xp[i, 0, 0] = x_pred
        s_xp[i, 1, 0] = y_pred
        s_xp[i, 2, 0] = th_pred

    # Predicted mean
    xp = np.mean(s_xp, axis=0)   # (3,1)
    xp[2,0] = wrap_angle(xp[2,0])

    # Predicted covariance Pp
    Pp = np.zeros((3,3))
    for i in range(num_sigma):
        dx = s_xp[i] - xp
        dx[2,0] = wrap_angle(dx[2,0])
        Pp += (dx @ dx.T) / num_sigma

    # If measurement is missing -> skip update
    if np.isnan(measurement_xy[0]) or np.isnan(measurement_xy[1]) or np.isnan(measurement_yaw):
        x = xp[0,0]
        y = xp[1,0]
        theta = xp[2,0]
        Pm = Pp
        internalStateOut = [x, y, theta, Pm]
        return x, y, theta, internalStateOut

    # ----- Measurement prediction -----
    s_z = np.zeros((num_sigma, 3, 1))  # [x_meas, y_meas, yaw_meas]

    for i in range(num_sigma):
        # measurement noise from augmented sigma points
        n_x   = s_new[i, 5, 0]
        n_y   = s_new[i, 6, 0]
        n_yaw = s_new[i, 7, 0]

        # predicted measurement = state + noise
        #d = 0.5 * B  # distance from rear axle state to car center
        #x_center = s_xp[i, 0, 0] + d * np.cos(s_xp[i, 2, 0])
        #y_center = s_xp[i, 1, 0] + d * np.sin(s_xp[i, 2, 0])
        #z_x   = x_center + n_x
        #z_y   = y_center + n_y

        z_x   = s_xp[i, 0, 0] + n_x
        z_y   = s_xp[i, 1, 0] + n_y
        z_yaw = s_xp[i, 2, 0] + n_yaw
        z_yaw = wrap_angle(z_yaw)

        s_z[i, 0, 0] = z_x
        s_z[i, 1, 0] = z_y
        s_z[i, 2, 0] = z_yaw

    # Mean of predicted measurement
    zp = np.mean(s_z, axis=0)  # (3,1)
    zp[2,0] = wrap_angle(zp[2,0])

    # Measurement covariance Pzz
    Pzz = np.zeros((3,3))
    for i in range(num_sigma):
        dz = s_z[i] - zp
        dz[2,0] = wrap_angle(dz[2,0])
        Pzz += (dz @ dz.T) / num_sigma

    # Cross covariance Pxz
    Pxz = np.zeros((3,3))
    for i in range(num_sigma):
        dx = s_xp[i] - xp
        dx[2,0] = wrap_angle(dx[2,0])
        dz = s_z[i] - zp
        dz[2,0] = wrap_angle(dz[2,0])
        Pxz += (dx @ dz.T) / num_sigma

    # Kalman gain
    K = Pxz @ np.linalg.inv(Pzz)  # (3x3)

    # Actual measurement vector
    z = np.array([
        [measurement_xy[0]],
        [measurement_xy[1]],
        [measurement_yaw]
    ])

    y_tilde = z - zp
    y_tilde[2,0] = wrap_angle(y_tilde[2,0])

    xm = xp + K @ y_tilde
    xm[2,0] = wrap_angle(xm[2,0])

    Pm = Pp - K @ Pzz @ K.T

    x = xm[0,0]
    y = xm[1,0]
    theta = xm[2,0]

    internalStateOut = [x, y, theta, Pm]
    return x, y, theta, internalStateOut


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
        experimentalData[0,2],   # x0
        experimentalData[0,3],   # y0
        experimentalData[0,4]    # yaw0
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
            dt = experimentalData[k,0] - experimentalData[k-1,0]

        t       = experimentalData[k,0]
        measx   = experimentalData[k,2]
        measy   = experimentalData[k,3]
        measyaw = experimentalData[k,4]
        speed   = experimentalData[k,5]
        gamma   = experimentalData[k,6]

        x, y, theta, internalState = estRun(
            t, dt, internalState, gamma, speed, (measx, measy), measyaw
        )

        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k]      = theta

    print('Done running')

    estimatedAngle = wrap_angle(estimatedAngle)

    true_x   = experimentalData[:,2]
    true_y   = experimentalData[:,3]
    true_yaw = experimentalData[:,4]

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

    t_vec = experimentalData[:,0]

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

    axHist[3].plot(t_vec, experimentalData[:,6], 'g-', label='steering')
    axHist[4].plot(t_vec, experimentalData[:,5], 'g-', label='speed')

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
