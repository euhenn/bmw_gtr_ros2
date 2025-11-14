import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# ---------- Bicycle Model ----------
B = 0.267  # wheelbase [m]


def fx(state, dt, u):
    """Process model for the bicycle model."""
    x, y, theta = state
    v, gamma = u

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(gamma) * dt / B
    theta_new = wrap_angle(theta_new)

    return np.array([x_new, y_new, theta_new])


def hx_full(state):
    """Measurement model for GPS + yaw"""
    x, y, theta = state
    return np.array([x, y, theta])


def hx_yaw(state):
    """Measurement model for yaw-only measurement"""
    return np.array([state[2]])


# ---------- Angle-aware mean/residual functions ----------
def state_mean(sigmas, Wm):
    x = np.zeros(3)
    x[0] = np.dot(Wm, sigmas[:, 0])
    x[1] = np.dot(Wm, sigmas[:, 1])
    x[2] = np.arctan2(np.dot(Wm, np.sin(sigmas[:, 2])),
                      np.dot(Wm, np.cos(sigmas[:, 2])))
    return x


def meas_mean(sigmas, Wm):
    z = np.zeros(3)
    z[0] = np.dot(Wm, sigmas[:, 0])
    z[1] = np.dot(Wm, sigmas[:, 1])
    z[2] = np.arctan2(np.dot(Wm, np.sin(sigmas[:, 2])),
                      np.dot(Wm, np.cos(sigmas[:, 2])))
    return z


def residual_x(a, b):
    y = a - b
    y[2] = wrap_angle(y[2])
    return y


def residual_z(a, b):
    y = a - b
    y[2] = wrap_angle(y[2])
    return y


# ============================================================================
#   UKF CLASS
# ============================================================================
class UKFEstimator:
    def __init__(self, x0, y0, yaw0):
        """Initialize the UKF with first measurements."""

        # Sigma points
        points = MerweScaledSigmaPoints(
            n=3,
            alpha=0.0001,
            beta=2.0,
            kappa=0.0
        )

        # UKF instance
        self.ukf = UnscentedKalmanFilter(
            dim_x=3,
            dim_z=3,
            dt=0.01,
            hx=hx_full,
            fx=fx,
            points=points,
            x_mean_fn=state_mean,
            z_mean_fn=meas_mean,
            residual_x=residual_x,
            residual_z=residual_z
        )

        # Initial state
        self.ukf.x = np.array([x0, y0, yaw0])

        # Initial covariance
        self.ukf.P = np.diag([
            0.05**2,                 # x
            0.05**2,                 # y
            (5*np.pi/180.0)**2       # yaw
        ])

        # Process noise (model uncertainty)
        self.ukf.Q = np.diag([
            0.01**2,
            0.01**2,
            (2*np.pi/180.0)**2
        ])

        # Measurement noise for full GPS+yaw update
        sigma_x = 1e-3
        sigma_y = 1e-3
        sigma_yaw = 1 * np.pi / 180
        self.R_full = np.diag([sigma_x**2, sigma_y**2, sigma_yaw**2])

        # Measurement noise for yaw-only IMU update
        self.R_yaw = np.array([[sigma_yaw**2]])

    # ----------------------------------------------------------------------
    #   PREDICTION STEP (run at IMU frequency)
    # ----------------------------------------------------------------------
    def predict(self, dt, speed, steering):
        u = np.array([speed, steering])
        self.ukf.predict(dt=dt, u=u)

    # ----------------------------------------------------------------------
    #   GPS + YAW UPDATE  (low rate)
    # ----------------------------------------------------------------------
    def update_gps_yaw(self, x_meas, y_meas, yaw_meas):
        z = np.array([x_meas, y_meas, yaw_meas])

        # temporarily replace measurement noise
        old_R = self.ukf.R
        self.ukf.R = self.R_full

        self.ukf.update(z)

        self.ukf.R = old_R  # restore

    # ----------------------------------------------------------------------
    #   YAW-ONLY UPDATE  (IMU rate)
    # ----------------------------------------------------------------------
    def update_yaw(self, yaw_meas):
        z = np.array([yaw_meas])

        # yaw-only update uses hx_yaw, R_yaw
        self.ukf.update(
            z,
            hx=hx_yaw,
            R=self.R_yaw
        )

    # ----------------------------------------------------------------------
    #   High-level update (handles missing sensors)
    # ----------------------------------------------------------------------
    def step(self, dt, speed, steering, meas_x, meas_y, meas_yaw):
        """Call at IMU frequency. GPS can be slower."""

        # 1) Prediction always
        self.predict(dt, speed, steering)

        gps_ok = not (np.isnan(meas_x) or np.isnan(meas_y))
        yaw_ok = not np.isnan(meas_yaw)

        # 2) No sensors → prediction only
        if not gps_ok and not yaw_ok:
            return self.state()

        # 3) Only yaw available → IMU update
        if yaw_ok and not gps_ok:
            self.update_yaw(meas_yaw)
            return self.state()

        # 4) GPS + yaw available
        if gps_ok and yaw_ok:
            self.update_gps_yaw(meas_x, meas_y, meas_yaw)
            return self.state()

        # 5) GPS only (rare, but handle)
        if gps_ok and not yaw_ok:
            # treat yaw as unobserved
            self.update_gps_yaw(meas_x, meas_y, np.nan)
            return self.state()

    # ----------------------------------------------------------------------
    # Helper
    # ----------------------------------------------------------------------
    def state(self):
        return self.ukf.x[0], self.ukf.x[1], self.ukf.x[2]
