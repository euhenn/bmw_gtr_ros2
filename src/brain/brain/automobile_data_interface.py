#!/usr/bin/python3
import numpy as np
import time
from collections import deque

# === CONSTANTS ===
YAW_GLOBAL_OFFSET = np.deg2rad(0)
START_X, START_Y = 0.2, 14.8

# Sensor delays and frequencies
GPS_DELAY = 0.45       # [s]
ENCODER_POS_FREQ = 100.0
GPS_FREQ = 10.0
BUFFER_PAST_MEASUREMENTS_LENGTH = int(round(GPS_DELAY * ENCODER_POS_FREQ))

# Vehicle limits
MIN_SPEED, MAX_SPEED = -0.3, 2.5      # [m/s]
MAX_ACCEL, MAX_STEER = 5.5, 28.0      # [m/s²], [deg]

# Vehicle geometry
LENGTH, WIDTH = 0.45, 0.18
BACKTOWHEEL, WHEEL_LEN, WHEEL_WIDTH = 0.10, 0.03, 0.03
WB = 0.04 #0.26

# Camera parameters
FRAME_WIDTH, FRAME_HEIGHT = 320, 240
CAM_X, CAM_Y, CAM_Z = 0.0, 0.0, 0.2
CAM_ROLL, CAM_PITCH, CAM_YAW = 0.0, np.deg2rad(20), 0.0
CAM_FOV, CAM_F = 1.085594795, 1.0
CAM_Sx, CAM_Sy, CAM_Ox, CAM_Oy = 10.0, 10.0, 10.0, 10.0
CAM_K = np.array([
    [CAM_F * CAM_Sx, 0.0, CAM_Ox],
    [0.0, CAM_F * CAM_Sy, CAM_Oy],
    [0.0, 0.0, 1.0]
])

# Estimator
EST_INIT_X, EST_INIT_Y, EST_INIT_YAW = 3.0, 3.0, 0.0
EKF_STEPS_BEFORE_TRUST = 10


# === CLASS DEFINITION ===
class Automobile_Data:
    """Base class for automobile data representation."""

    def __init__(self):
        # --- Position ---
        self.x_true, self.y_true = START_X, START_Y
        self.x = self.y = 0.0

        # --- IMU ---
        self.yaw_offset = YAW_GLOBAL_OFFSET
        self.roll = self.pitch = self.yaw = 0.0
        self.roll_deg = self.pitch_deg = self.yaw_deg = 0.0
        self.yaw_true = 0.0
        self.accel_x = self.accel_y = self.accel_z = 0.0
        self.gyrox = self.gyroy = self.gyroz = 0.0
        self.filtered_yaw = self.IMU_yaw = 0.0

        # --- Encoder ---
        self.encoder_velocity = self.filtered_encoder_velocity = 0.0
        self.encoder_distance = self.prev_dist = self.prev_gps_dist = 0.0

        # --- Estimation ---
        self.x_est = self.y_est = 0.0
        self.yaw_est = self.yaw_offset
        self.gps_cnt = 0
        self.trust_gps = True
        self.buffer_gps_positions_still_car = []

        # --- Local coordinates ---
        self.x_loc = self.y_loc = self.yaw_loc = 0.0
        self.yaw_loc_o = self.dist_loc = self.dist_loc_o = 0.0
        self.last_gps_sample_time = time.time()
        self.new_gps_sample_arrived = True

        # --- Sensors ---
        self.sonar_distance = self.filtered_sonar_distance = 3.0
        self.right_sonar_distance = self.filtered_right_sonar_distance = 3.0
        self.left_sonar_distance = self.filtered_left_sonar_distance = 3.0
        self.center_tof_distance = self.filtered_center_tof_distance = 0.21
        self.left_tof_distance = self.filtered_left_tof_distance = 0.21
        self.obstacle = self.filtered_obstacle = 0.0
        self.sign = self.filtered_sign = 0.0

        # --- Camera ---
        self.frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3), np.uint8)

        # --- Control ---
        self.speed = 0.0
        self.steer = 0.0

        # --- Constants ---
        self.MIN_SPEED = MIN_SPEED
        self.MAX_SPEED = MAX_SPEED
        self.MAX_STEER = MAX_STEER
        self.LENGTH, self.WIDTH, self.WB = LENGTH, WIDTH, WB
        self.CAM_K = CAM_K

        # --- Buffers ---
        self.past_encoder_distances = deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.past_yaws = deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.yaws_between_updates = deque(maxlen=int(round(ENCODER_POS_FREQ / GPS_FREQ)))

        # --- EKF init state ---
        self.est_init_state = np.array([EST_INIT_X, EST_INIT_Y]).reshape(-1, 1)
        self.STARTED_WITH_IMU = False

    # === Methods ===
    def drive(self, speed=0.0, angle=0.0):
        self.speed = self.normalizeSpeed(speed)
        self.steer = self.normalizeSteer(angle)

    def update_rel_position(self):
        """Update local frame pose."""
        self.yaw_loc = self.yaw - self.yaw_loc_o
        curr_dist = self.encoder_distance
        self.past_encoder_distances.append(curr_dist)
        if len(self.past_yaws) >= BUFFER_PAST_MEASUREMENTS_LENGTH:
            self.yaws_between_updates.append(self.past_yaws.popleft())
        self.past_yaws.append(self.yaw)

        L = np.abs(curr_dist - self.prev_dist)
        self.x_loc += L * np.cos(self.yaw_loc)
        self.y_loc += L * np.sin(self.yaw_loc)
        self.prev_dist = curr_dist

    def reset_rel_pose(self):
        """Reset local frame origin."""
        self.x_loc = self.y_loc = self.yaw_loc = 0.0
        self.yaw_loc_o = self.yaw
        self.dist_loc_o = self.encoder_distance

    @staticmethod
    def normalizeSpeed(val):
        return max(MIN_SPEED, min(val, MAX_SPEED))

    @staticmethod
    def normalizeSteer(val):
        return max(-MAX_STEER, min(val, MAX_STEER))

    def __str__(self):
        return f"""
{'='*60}
GLOBAL POSITION: ({self.x:.2f}, {self.y:.2f})
LOCAL POSITION:  ({self.x_loc:.2f}, {self.y_loc:.2f}) Yaw: {np.rad2deg(self.yaw_loc):.1f}°
ENCODER: dist={self.encoder_distance:.3f}m vel={self.encoder_velocity:.2f}m/s
SONAR: {self.sonar_distance:.2f}m | YAW: {np.rad2deg(self.yaw):.1f}°
{'='*60}
"""
