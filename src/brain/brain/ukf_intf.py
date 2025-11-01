#!/usr/bin/env python3
import numpy as np
import math
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


class CarUKF:
    """
    Unscented Kalman Filter for a 4-state kinematic bicycle model:
        x = [x, y, yaw, v]
    Control input:
        u = [a, delta]
    Sensors:
        - GPS: (x, y)
        - IMU: yaw
        - Encoder: v (and optionally steering for predict)
    """

    def __init__(self, wheelbase=0.26, dt=0.01):
        self.L = wheelbase
        self.dt = dt

        points = MerweScaledSigmaPoints(
            n=4, alpha=0.1, beta=2.0, kappa=0.0,
            subtract=self._residual_x
        )

        self.ukf = UKF(
            dim_x=4, dim_z=4,
            fx=self._fx,
            hx=self._hx_full,
            dt=self.dt,
            points=points,
            x_mean_fn=self._state_mean,
            z_mean_fn=self._z_mean,
            residual_x=self._residual_x,
            residual_z=self._residual_z,
        )

        # === Initialization ===
        self.ukf.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.ukf.P = np.diag([0.5, 0.5, np.deg2rad(5)**2, 0.2])
        self.ukf.Q = np.diag([1e-3, 1e-3, 1e-4, 2e-3])

        # Measurement noise
        self.SIGMA_GPS_XY = 0.15
        self.SIGMA_IMU_YAW = np.deg2rad(1.0)
        self.SIGMA_V = 0.05

        self.R_full = np.diag([
            self.SIGMA_GPS_XY**2,
            self.SIGMA_GPS_XY**2,
            self.SIGMA_IMU_YAW**2,
            self.SIGMA_V**2
        ])

        # Internal memory
        self._prev_v = 0.0

    # ========================================
    # Motion model
    # ========================================
    def _fx(self, x, dt, u, L):
        px, py, yaw, v = x
        a, delta = u
        px += v * math.cos(yaw) * dt
        py += v * math.sin(yaw) * dt
        yaw += (v / L) * math.tan(delta) * dt
        v += a * dt
        yaw = self._wrap_angle(yaw)
        return np.array([px, py, yaw, v])

    def _hx_full(self, x):
        return np.array([x[0], x[1], x[2], x[3]])

    # ========================================
    # Angle helpers
    # ========================================
    def _wrap_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _residual_x(self, a, b):
        y = a - b
        y[2] = self._wrap_angle(y[2])
        return y

    def _residual_z(self, a, b):
        y = a - b
        if len(y) >= 3:
            y[2] = self._wrap_angle(y[2])
        return y

    # ========================================
    # Mean functions
    # ========================================
    def _state_mean(self, sigmas, Wm):
        x = np.zeros(4)
        x[0] = np.dot(Wm, sigmas[:, 0])
        x[1] = np.dot(Wm, sigmas[:, 1])
        s = np.dot(Wm, np.sin(sigmas[:, 2]))
        c = np.dot(Wm, np.cos(sigmas[:, 2]))
        x[2] = math.atan2(s, c)
        x[3] = np.dot(Wm, sigmas[:, 3])
        return x

    def _z_mean(self, sigmas, Wm):
        zdim = sigmas.shape[1]
        z = np.zeros(zdim)
        if zdim >= 1:
            z[0] = np.dot(Wm, sigmas[:, 0])
        if zdim >= 2:
            z[1] = np.dot(Wm, sigmas[:, 1])
        if zdim >= 3:
            s = np.dot(Wm, np.sin(sigmas[:, 2]))
            c = np.dot(Wm, np.cos(sigmas[:, 2]))
            z[2] = math.atan2(s, c)
        if zdim >= 4:
            z[3] = np.dot(Wm, sigmas[:, 3])
        return z

    # ========================================
    # Core UKF interface
    # ========================================
    def predict(self, v_meas, steer_angle, dt=None):
        """Predict using current control inputs (v, steer)."""
        if dt is None:
            dt = self.dt
        a_est = (v_meas - self._prev_v) / max(dt, 1e-3)
        self._prev_v = v_meas
        delta = np.deg2rad(steer_angle)
        u = np.array([a_est, delta])
        try:
            self.ukf.predict(u=u, L=self.L, dt=dt)
        except TypeError:
            self.ukf.predict(u=u, L=self.L)

    # ========================================
    # Unified sensor update
    # ========================================
    def update_sensor(self, sensor_type, measurement):
        """
        Unified interface for sensor updates.
        sensor_type ∈ {'gps', 'imu', 'encoder'}
        measurement:
            - gps: (x, y)
            - imu: yaw (rad)
            - encoder: (v)
        """
        if sensor_type == 'gps':
            x_m, y_m = measurement
            z = np.array([x_m, y_m])
            def hx(x): return np.array([x[0], x[1]])
            R = np.diag([self.SIGMA_GPS_XY**2, self.SIGMA_GPS_XY**2])
            self.ukf.update(z, R=R, hx=hx)

        elif sensor_type == 'imu':
            yaw_m = measurement if np.isscalar(measurement) else measurement[0]
            z = np.array([yaw_m])
            def hx(x): return np.array([x[2]])
            def resid_z(a, b):
                y = a - b
                y[0] = self._wrap_angle(y[0])
                return y
            prev_rz = self.ukf.residual_z
            self.ukf.residual_z = resid_z
            R = np.array([[self.SIGMA_IMU_YAW**2]])
            self.ukf.update(z, R=R, hx=hx)
            self.ukf.residual_z = prev_rz

        elif sensor_type == 'encoder':
            v_m = measurement if np.isscalar(measurement) else measurement[0]
            z = np.array([v_m])
            def hx(x): return np.array([x[3]])
            R = np.array([[self.SIGMA_V**2]])
            self.ukf.update(z, R=R, hx=hx)

        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    # ========================================
    # Accessors
    # ========================================
    @property
    def state(self):
        return self.ukf.x.copy()

    @property
    def covariance(self):
        return self.ukf.P.copy()

    def __str__(self):
        x, y, yaw, v = self.ukf.x
        return f"[UKF] x={x:.2f} y={y:.2f} yaw={np.rad2deg(yaw):.1f}° v={v:.2f} m/s"



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import sleep
    ukf = CarUKF()

    # simulation parameters
    dt = 0.01
    n_steps = 800
    true_x, true_y, true_yaw, true_v = 0.0, 0.0, 0.0, 1.0

    xs_est, ys_est, yaws_est, vs_est = [], [], [], []
    xs_true, ys_true, yaws_true, vs_true = [], [], [], []

    for i in range(n_steps):
        # --- Simulate true motion (straight line 1 m/s) ---
        true_x += true_v * np.cos(true_yaw) * dt
        true_y += true_v * np.sin(true_yaw) * dt
        true_yaw += 0.0
        true_yaw = (true_yaw + np.pi) % (2 * np.pi) - np.pi

        # --- UKF predict step ---
        ukf.predict(v_meas=true_v, steer_angle=0.0, dt=dt)

        # --- Simulate and apply noisy sensor updates ---
        if i % 10 == 0:  # GPS
            noisy_x = true_x + np.random.randn() * 0.1
            noisy_y = true_y + np.random.randn() * 0.1
            ukf.update_sensor('gps', (noisy_x, noisy_y))

        # IMU yaw update (every iteration)
        noisy_yaw = true_yaw + np.random.randn() * np.deg2rad(1)
        ukf.update_sensor('imu', noisy_yaw)

        # Encoder speed update (every iteration)
        noisy_v = true_v + np.random.randn() * 0.05
        ukf.update_sensor('encoder', noisy_v)

        # --- Record for plotting ---
        x, y, yaw, v = ukf.state
        xs_est.append(x)
        ys_est.append(y)
        yaws_est.append(yaw)
        vs_est.append(v)

        xs_true.append(true_x)
        ys_true.append(true_y)
        yaws_true.append(true_yaw)
        vs_true.append(true_v)

        print(ukf)
        sleep(dt)

    # ===========================
    # Plot results
    # ===========================
    t = np.arange(n_steps) * dt

    plt.figure(figsize=(10, 8))

    # XY position
    plt.subplot(3, 1, 1)
    plt.plot(xs_true, ys_true, 'k-', label="True path")
    plt.plot(xs_est, ys_est, 'r--', label="UKF estimated")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Position")
    plt.legend()
    plt.axis("equal")

    # Yaw
    plt.subplot(3, 1, 2)
    plt.plot(t, np.unwrap(yaws_true), 'k-', label="True yaw")
    plt.plot(t, np.unwrap(yaws_est), 'r--', label="UKF yaw")
    plt.ylabel("Yaw [rad]")
    plt.legend()

    # Velocity
    plt.subplot(3, 1, 3)
    plt.plot(t, vs_true, 'k-', label="True v")
    plt.plot(t, vs_est, 'r--', label="UKF v")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()

    plt.tight_layout()
    plt.show()
