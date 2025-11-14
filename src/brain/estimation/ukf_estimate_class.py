import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# ======================================================================
#                      UKF ESTIMATOR (8-DIM, ORIGINAL)
# ======================================================================
class UKF_Estimator_8D:
    """
    THIS IS A 1:1 PORT OF YOUR ORIGINAL UKF.
    Full augmentation:
      [x, y, theta, v_noise, gamma_noise, meas_x_noise, meas_y_noise, meas_yaw_noise]

    This class reproduces your original results exactly.
    """

    def __init__(self, B=0.267):
        self.B = B

        # ----- Process noise -----
        self.sigma_v_proc = 0.03
        self.sigma_gam_proc = 2*np.pi/180
        self.V = np.diag([
            self.sigma_v_proc**2,
            self.sigma_gam_proc**2
        ])
        self.V_mean = np.zeros((2,1))

        # ----- Measurement noise -----
        self.sigma_x_meas = 1e-3
        self.sigma_y_meas = 1e-3
        self.sigma_yaw_meas = 1*np.pi/180

        self.W = np.diag([
            self.sigma_x_meas**2,
            self.sigma_y_meas**2,
            self.sigma_yaw_meas**2
        ])
        self.W_mean = np.zeros((3,1))

        # Internal filter state
        self.x = None
        self.y = None
        self.th = None
        self.P = None
        self.last_time = None

    # ------------------------------------------------------------------
    def initialize(self, x0, y0, yaw0, t0):
        self.x = float(x0)
        self.y = float(y0)
        self.th = float(yaw0)
        self.last_time = float(t0)

        # same init covariance as original
        self.P = np.diag([
            0.05**2,
            0.05**2,
            (5*np.pi/180)**2
        ])

    # ------------------------------------------------------------------
    def step(self, time, steeringAngle, speed, gps_xy=None, imu_yaw=None):
        # -------- dt from timestamps --------
        time = float(time)
        if self.last_time is None:
            dt = 0.0
        else:
            dt = time - self.last_time
        self.last_time = time

        # -------- previous state --------
        x_last = self.x
        y_last = self.y
        th_last = self.th
        Pm = self.P

        x_state = np.array([[x_last],
                            [y_last],
                            [th_last]])

        # -------- build augmented state --------
        # [x, y, th, v_noise, g_noise, meas_x_noise, meas_y_noise, meas_yaw_noise]
        x_aug = np.vstack((x_state, self.V_mean, self.W_mean))  # (8x1)
        P_aug = scipy.linalg.block_diag(Pm, self.V, self.W)     # (8x8)

        # -------- sigma points --------
        n_aug = 8
        S = scipy.linalg.sqrtm(n_aug * P_aug)

        sigmas = np.zeros((2*n_aug, n_aug, 1))
        for i in range(n_aug):
            col = S[:, i:i+1]
            sigmas[i]       = x_aug + col
            sigmas[i+n_aug] = x_aug - col

        # -------- prediction --------
        s_xp = np.zeros((2*n_aug, 3, 1))

        for i in range(2*n_aug):
            x_i  = sigmas[i,0,0]
            y_i  = sigmas[i,1,0]
            th_i = sigmas[i,2,0]
            v_n  = sigmas[i,3,0]
            g_n  = sigmas[i,4,0]

            v_i = speed + v_n
            g_i = steeringAngle + g_n

            x_p  = x_i + v_i*np.cos(th_i)*dt
            y_p  = y_i + v_i*np.sin(th_i)*dt
            th_p = th_i + v_i*np.tan(g_i)*dt/self.B
            th_p = wrap_angle(th_p)

            s_xp[i,0,0] = x_p
            s_xp[i,1,0] = y_p
            s_xp[i,2,0] = th_p

        # predicted mean
        xp = np.mean(s_xp, axis=0)
        xp[2,0] = wrap_angle(xp[2,0])

        # predicted covariance
        Pp = np.zeros((3,3))
        for i in range(2*n_aug):
            dx = s_xp[i] - xp
            dx[2,0] = wrap_angle(dx[2,0])
            Pp += (dx @ dx.T) / (2*n_aug)

        # ------------------------------------------------------------------
        # If ANY measurement is missing → skip update (pure prediction)
        # ------------------------------------------------------------------
        if gps_xy is None or imu_yaw is None:
            self.x, self.y, self.th = xp[0,0], xp[1,0], xp[2,0]
            self.P = Pp
            return self.x, self.y, self.th, self.P

        mx, my = gps_xy
        myaw = imu_yaw

        if np.isnan(mx) or np.isnan(my) or np.isnan(myaw):
            self.x, self.y, self.th = xp[0,0], xp[1,0], xp[2,0]
            self.P = Pp
            return self.x, self.y, self.th, self.P

        # ------------------------------------------------------------------
        # Measurement prediction (3-dim)
        # ------------------------------------------------------------------
        s_z = np.zeros((2*n_aug, 3, 1))

        for i in range(2*n_aug):
            n_x = sigmas[i,5,0]
            n_y = sigmas[i,6,0]
            n_yaw = sigmas[i,7,0]

            z_x = s_xp[i,0,0] + n_x
            z_y = s_xp[i,1,0] + n_y
            z_th = wrap_angle(s_xp[i,2,0] + n_yaw)

            s_z[i,0,0] = z_x
            s_z[i,1,0] = z_y
            s_z[i,2,0] = z_th

        zp = np.mean(s_z, axis=0)
        zp[2,0] = wrap_angle(zp[2,0])

        # Pzz and Pxz
        Pzz = np.zeros((3,3))
        Pxz = np.zeros((3,3))

        for i in range(2*n_aug):
            dz = s_z[i] - zp
            dz[2,0] = wrap_angle(dz[2,0])

            dx = s_xp[i] - xp
            dx[2,0] = wrap_angle(dx[2,0])

            Pzz += (dz @ dz.T) / (2*n_aug)
            Pxz += (dx @ dz.T) / (2*n_aug)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        # measurement
        z = np.array([[mx],[my],[myaw]])
        y_tilde = z - zp
        y_tilde[2,0] = wrap_angle(y_tilde[2,0])

        xm = xp + K @ y_tilde
        xm[2,0] = wrap_angle(xm[2,0])

        Pm = Pp - K @ Pzz @ K.T

        # update internal state
        self.x = xm[0,0]
        self.y = xm[1,0]
        self.th = xm[2,0]
        self.P = Pm

        return self.x, self.y, self.th, self.P


# ======================================================================
#                                 MAIN
# ======================================================================
if __name__ == "__main__":

    experimentalRun = 1

    print("Loading data...")
    data = np.genfromtxt(
        f"data/mpc_{experimentalRun:03d}.csv",
        delimiter=",",
        skip_header=1
    )

    t_vec  = data[:,0]
    gps_x  = data[:,2]
    gps_y  = data[:,3]
    imu_yaw = data[:,4]
    speed  = data[:,5]
    gamma  = data[:,6]

    N = len(data)

    # init UKF
    ukf = UKF_Estimator_8D()
    ukf.initialize(gps_x[0], gps_y[0], imu_yaw[0], t_vec[0])

    est_x = np.zeros(N)
    est_y = np.zeros(N)
    est_th = np.zeros(N)

    print("Running UKF...")
    for k in range(N):
        gps_xy = (gps_x[k], gps_y[k])
        yaw_meas = imu_yaw[k]

        x, y, th, _ = ukf.step(
            time=t_vec[k],
            steeringAngle=gamma[k],
            speed=speed[k],
            gps_xy=gps_xy,
            imu_yaw=yaw_meas
        )

        est_x[k] = x
        est_y[k] = y
        est_th[k] = th

    print("UKF done.")

    # Errors
    err_x = est_x - gps_x
    err_y = est_y - gps_y
    err_th = wrap_angle(est_th - imu_yaw)

    print("Final error:")
    print("   x:", err_x[-1])
    print("   y:", err_y[-1])
    print("   yaw:", err_th[-1])

    # ==================================================================
    #                                PLOTS
    # ==================================================================
    print("Plotting...")

    # --- Top View ---
    fig1, ax1 = plt.subplots()
    ax1.plot(gps_x, gps_y, "k:", label="measured")
    ax1.plot(est_x, est_y, "b-", label="UKF estimate")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Top View")
    ax1.legend()

    # --- Time Histories ---
    fig2, axs = plt.subplots(5, 1, sharex=True)

    axs[0].plot(t_vec, gps_x, 'k:', label='meas x')
    axs[0].plot(t_vec, est_x, 'b-', label='est x')
    axs[0].legend()

    axs[1].plot(t_vec, gps_y, 'k:', label='meas y')
    axs[1].plot(t_vec, est_y, 'b-', label='est y')

    axs[2].plot(t_vec, imu_yaw, 'k:', label='meas yaw')
    axs[2].plot(t_vec, est_th, 'b-', label='est yaw')

    axs[3].plot(t_vec, gamma, 'g-', label='steering')
    axs[4].plot(t_vec, speed, 'g-', label='speed')

    axs[4].set_xlabel("Time [s]")
    axs[0].set_ylabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[2].set_ylabel("yaw [rad]")
    axs[3].set_ylabel("δ [rad]")
    axs[4].set_ylabel("v [m/s]")

    # --- Error plots ---
    fig3, ax3 = plt.subplots(3, 1, sharex=True)
    ax3[0].plot(t_vec, err_x, 'r-', label='x error')
    ax3[1].plot(t_vec, err_y, 'r-', label='y error')
    ax3[2].plot(t_vec, err_th, 'r-', label='yaw error')

    ax3[2].set_xlabel("Time [s]")
    ax3[0].set_ylabel("x err [m]")
    ax3[1].set_ylabel("y err [m]")
    ax3[2].set_ylabel("yaw err [rad]")

    plt.show()
