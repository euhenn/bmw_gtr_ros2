#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from bmw_gtr.msg import IMU, Localisation
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import collections
import numpy as np
from time import time
import helper_functions as hf
from automobile_data_interface import Automobile_Data

REALISTIC = False

ENCODER_TIMER = 0.01  # frequency of encoder reading
STEER_UPDATE_FREQ = 50.0 if REALISTIC else 150.0
SERVO_DEAD_TIME_DELAY = 0.15 if REALISTIC else 0.0
MAX_SERVO_ANGULAR_VELOCITY = 2.8 if REALISTIC else 15.0
DELTA_ANGLE = np.rad2deg(MAX_SERVO_ANGULAR_VELOCITY) / STEER_UPDATE_FREQ
MAX_STEER_COMMAND_FREQ = 50.0
MAX_STEER_SAMPLES = max(int((2 * SERVO_DEAD_TIME_DELAY) * MAX_STEER_COMMAND_FREQ), 10)

GPS_DELAY = 0.45  # [s] delay for gps message to arrive
ENCODER_POS_FREQ = 100.0  # [Hz] frequency of encoder position messages
GPS_FREQ = 10.0  # [Hz] frequency of gps messages
BUFFER_PAST_MEASUREMENTS_LENGTH = int(round(GPS_DELAY * ENCODER_POS_FREQ))


class AutomobileDataSimulator(Node):
    def __init__(self,
                 trig_control=True,
                 trig_bno=False,
                 trig_enc=False,
                 trig_cam=False,
                 trig_gps=False):

        # Initialize Node first
        super().__init__('automobile_data_simulator')
        
        # Initialize all Automobile_Data attributes manually
        self._init_automobile_data()

        # ADDITIONAL VARIABLES
        self.timestamp = 0.0
        self.prev_x_true = self.x_true
        self.prev_y_true = self.y_true
        self.prev_timestamp = 0.0
        self.velocity_buffer = collections.deque(maxlen=20)
        self.target_steer = 0.0
        self.curr_steer = 0.0
        self.steer_deque = collections.deque(maxlen=MAX_STEER_SAMPLES)
        self.time_last_steer_command = time()
        self.target_dist = 0.0
        self.arrived_at_dist = True
        self.yaw_true = 0.0

        self.x_buffer = collections.deque(maxlen=5)
        self.y_buffer = collections.deque(maxlen=5)

        # PUBLISHERS AND SUBSCRIBERS
        if trig_control:
            self.pub = self.create_publisher(String, '/automobile/command', 10)
            self.steer_updater = self.create_timer(1.0 / STEER_UPDATE_FREQ, self.steer_update_callback)
            self.drive_dist_updater = self.create_timer(ENCODER_TIMER, self.drive_distance_callback)

        if trig_bno:
            self.sub_imu = self.create_subscription(IMU, '/automobile/IMU', self.imu_callback, 10)

        if trig_enc:
            self.reset_rel_pose()
            self.create_timer(ENCODER_TIMER, self.encoder_distance_callback)

        if trig_cam:
            self.bridge = CvBridge()
            self.sub_cam = self.create_subscription(Image, "/automobile/camera/image_raw", self.camera_callback, 10)

        if trig_gps:
            self.sub_pos = self.create_subscription(Localisation, "/automobile/localisation", self.position_callback, 10)

    def _init_automobile_data(self):
        """Initialize all attributes from Automobile_Data class"""
        # Import constants from automobile_data_interface
        from automobile_data_interface import (
            START_X, START_Y, YAW_GLOBAL_OFFSET, GPS_DELAY, ENCODER_POS_FREQ, GPS_FREQ,
            BUFFER_PAST_MEASUREMENTS_LENGTH, MIN_SPEED, MAX_SPEED, MAX_ACCEL, MAX_STEER,
            LENGTH, WIDTH, BACKTOWHEEL, WHEEL_LEN, WHEEL_WIDTH, WB,
            FRAME_WIDTH, FRAME_HEIGHT, CAM_X, CAM_Y, CAM_Z, CAM_ROLL, CAM_PITCH, 
            CAM_YAW, CAM_FOV, CAM_F, CAM_Sx, CAM_Sy, CAM_Ox, CAM_Oy, CAM_K,
            EST_INIT_X, EST_INIT_Y, EST_INIT_YAW, EKF_STEPS_BEFORE_TRUST
        )
        
        # CAR POSITION
        self.x_true = START_X
        self.y_true = START_Y
        self.x = 0.0
        self.y = 0.0
        
        # IMU
        self.yaw_offset = YAW_GLOBAL_OFFSET
        self.roll = 0.0
        self.roll_deg = 0.0
        self.pitch = 0.0
        self.pitch_deg = 0.0
        self.yaw = 0.0
        self.yaw_deg = 0.0
        self.yaw_true = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.gyrox = 0.0
        self.gyroy = 0.0
        self.gyroz = 0.0
        self.filtered_yaw = 0.0
        self.yaw_random_start = 0.0
        self.IMU_yaw = 0.0
        
        # ENCODER
        self.encoder_velocity = 0.0
        self.filtered_encoder_velocity = 0.0
        self.encoder_distance = 0.0
        self.prev_dist = 0.0
        self.prev_gps_dist = 0.0
        
        # CAR POSE ESTIMATION
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw_est = self.yaw_offset
        self.gps_cnt = 0
        self.trust_gps = True
        self.buffer_gps_positions_still_car = []
        
        # LOCAL POSITION
        self.x_loc = 0.0
        self.y_loc = 0.0
        self.yaw_loc = 0.0
        self.yaw_loc_o = 0.0
        self.dist_loc = 0.0
        self.dist_loc_o = 0.0
        self.last_gps_sample_time = time()
        self.new_gps_sample_arrived = True
        
        # SONARs
        self.sonar_distance = 3.0
        self.filtered_sonar_distance = 3.0
        self.right_sonar_distance = 3.0
        self.filtered_right_sonar_distance = 3.0
        self.left_sonar_distance = 3.0
        self.filtered_left_sonar_distance = 3.0
        
        # TOFs
        self.center_tof_distance = 0.21
        self.filtered_center_tof_distance = 0.21
        self.left_tof_distance = 0.21
        self.filtered_left_tof_distance = 0.21
        
        # ESP32 CAMERA
        self.obstacle = 0.0
        self.filtered_obstacle = 0.0
        self.sign = 0.0
        self.filtered_sign = 0.0
        
        # CAMERA
        self.frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3), np.uint8)
        
        # CONTROL ACTION
        self.speed = 0.0
        self.steer = 0.0
        
        # CONSTANT PARAMETERS
        self.MIN_SPEED = MIN_SPEED
        self.MAX_SPEED = MAX_SPEED
        self.MAX_STEER = MAX_STEER
        
        # VEHICLE PARAMETERS
        self.LENGTH = LENGTH
        self.WIDTH = WIDTH
        self.BACKTOWHEEL = BACKTOWHEEL
        self.WHEEL_LEN = WHEEL_LEN
        self.WHEEL_WIDTH = WHEEL_WIDTH
        self.WB = WB
        
        # CAMERA PARAMETERS
        self.FRAME_WIDTH = FRAME_WIDTH
        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.CAM_X = CAM_X
        self.CAM_Y = CAM_Y
        self.CAM_Z = CAM_Z
        self.CAM_ROLL = CAM_ROLL
        self.CAM_PITCH = CAM_PITCH
        self.CAM_YAW = CAM_YAW
        self.CAM_FOV = CAM_FOV
        self.CAM_K = CAM_K
        
        # ESTIMATION PARAMETERS
        self.last_estimation_callback_time = None
        self.est_init_state = np.array([EST_INIT_X, EST_INIT_Y]).reshape(-1, 1)
        
        self.past_encoder_distances = collections.deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.past_yaws = collections.deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.yaws_between_updates = collections.deque(maxlen=int(round(ENCODER_POS_FREQ/GPS_FREQ)))
        
        self.STARTED_WITH_IMU = False

    # Copy all methods from Automobile_Data class
    def drive(self, speed=0.0, angle=0.0):
        """Command a speed and steer angle to the car"""
        self.drive_speed(speed)
        self.drive_angle(angle)

    def update_rel_position(self):
        """Update relative pose of the car"""
        self.yaw_loc = self.yaw - self.yaw_loc_o
        curr_dist = self.encoder_distance

        self.past_encoder_distances.append(curr_dist)
        if len(self.past_yaws) > BUFFER_PAST_MEASUREMENTS_LENGTH-1:
            self.yaws_between_updates.append(self.past_yaws.popleft())
        self.past_yaws.append(self.yaw)

        self.dist_loc = np.abs(curr_dist - self.dist_loc_o)
        signed_L = curr_dist - self.prev_dist
        L = np.abs(signed_L)
        dx = L * np.cos(self.yaw_loc)
        dy = L * np.sin(self.yaw_loc)
        self.x_loc += dx
        self.y_loc += dy
        self.prev_dist = curr_dist
        
        if self.new_gps_sample_arrived:
            self.last_gps_sample_time = time()
            self.new_gps_sample_arrived = False

    def reset_rel_pose(self):
        """Set origin of the local frame to the actual pose"""
        self.x_loc = 0.0
        self.y_loc = 0.0
        self.yaw_loc_o = self.yaw
        self.prev_yaw = self.yaw
        self.yaw_loc = 0.0
        self.prev_yaw_loc = 0.0
        self.dist_loc = 0.0
        self.dist_loc_o = self.encoder_distance

    @staticmethod
    def normalizeSpeed(val):
        """Clamp speed value"""
        from automobile_data_interface import MIN_SPEED, MAX_SPEED
        if val < MIN_SPEED:
            val = MIN_SPEED
        elif val > MAX_SPEED:
            val = MAX_SPEED
        return val

    @staticmethod
    def normalizeSteer(val):
        """Clamp steer value"""
        from automobile_data_interface import MAX_STEER
        if val < -MAX_STEER:
            val = -MAX_STEER
        elif val > MAX_STEER:
            val = MAX_STEER
        return val

    # === CALLBACKS ===
    def camera_callback(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def position_callback(self, data):
        pL = np.array([data.pos_x, data.pos_y])
        pR = hf.mL2mR(pL)
        tmp_x = pR[0] - self.WB / 2 * np.cos(self.yaw)
        tmp_y = pR[1] - self.WB / 2 * np.sin(self.yaw)
        self.x_buffer.append(tmp_x)
        self.y_buffer.append(tmp_y)
        self.x = np.mean(self.x_buffer)
        self.y = np.mean(self.y_buffer)
        self.x_est = self.x
        self.y_est = self.y

    def imu_callback(self, data):
        self.roll = float(data.roll)
        self.roll_deg = np.rad2deg(self.roll)
        self.pitch = float(data.pitch)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_true = float(data.yaw)
        self.yaw = float(data.yaw) + self.yaw_offset
        self.yaw_deg = np.rad2deg(self.yaw)
        true_posL = np.array([data.posx, data.posy])
        true_posR = hf.mL2mR(true_posL)
        self.x_true = true_posR[0] - self.WB / 2 * np.cos(self.yaw_true)
        self.y_true = true_posR[1] - self.WB / 2 * np.sin(self.yaw_true)
        self.timestamp = float(data.timestamp)

    def encoder_distance_callback(self):
        curr_x = self.x_true
        curr_y = self.y_true
        prev_x = self.prev_x_true
        prev_y = self.prev_y_true
        curr_time = self.timestamp
        prev_time = self.prev_timestamp
        delta = np.hypot(curr_x - prev_x, curr_y - prev_y)
        motion_yaw = +np.arctan2(curr_y - prev_y, curr_x - prev_x)
        abs_yaw_diff = np.abs(hf.diff_angle(motion_yaw, self.yaw_true))
        sign = 1.0 if abs_yaw_diff < np.pi / 2 else -1.0
        dt = curr_time - prev_time
        if dt > 0.0:
            velocity = (delta * sign) / dt
            self.encoder_velocity_callback(velocity)
            self.encoder_distance += sign * delta
            self.prev_x_true = curr_x
            self.prev_y_true = curr_y
            self.prev_timestamp = curr_time
            self.update_rel_position()

    def steer_update_callback(self):
        if len(self.steer_deque) > 0:
            curr_time = time()
            angle, t = self.steer_deque.popleft()
            if curr_time - t < SERVO_DEAD_TIME_DELAY:
                self.steer_deque.appendleft((angle, t))
            else:
                self.target_steer = angle
        diff = self.target_steer - self.curr_steer
        if diff > 0.0:
            incr = min(diff, DELTA_ANGLE)
        elif diff < 0.0:
            incr = max(diff, -DELTA_ANGLE)
        else:
            return
        self.curr_steer += incr
        self.pub_steer(self.curr_steer)

    def encoder_velocity_callback(self, data):
        self.encoder_velocity = data
        self.velocity_buffer.append(self.encoder_velocity)
        self.filtered_encoder_velocity = np.median(self.velocity_buffer)

    # === COMMAND ACTIONS ===
    def drive_speed(self, speed=0.0):
        self.arrived_at_dist = True
        self.pub_speed(speed)

    def drive_angle(self, angle=0.0, direct=False):
        angle = self.normalizeSteer(angle)
        curr_time = time()
        if curr_time - self.time_last_steer_command > 1 / MAX_STEER_COMMAND_FREQ:
            self.time_last_steer_command = curr_time
            self.steer_deque.append((angle, curr_time))
        else:
            self.get_logger().warn('Missed steer command...')

    def drive_distance(self, dist=0.0):
        self.target_dist = self.encoder_distance + dist
        self.arrived_at_dist = False

    def drive_distance_callback(self):
        Kp = 0.5
        max_speed = 0.2
        if not self.arrived_at_dist:
            dist_error = self.target_dist - self.encoder_distance
            self.pub_speed(min(Kp * dist_error, max_speed))
            if np.abs(dist_error) < 0.01:
                self.arrived_at_dist = True
                self.drive_speed(0.0)

    def stop(self, angle=0.0):
        self.steer_deque.append((angle, time()))
        self.speed = 0.0
        data = {'action': '3', 'steerAngle': float(angle)}
        reference = json.dumps(data)
        msg = String()
        msg.data = reference
        self.pub.publish(msg)

    def pub_steer(self, angle):
        data = {'action': '2', 'steerAngle': float(angle)}
        reference = json.dumps(data)
        msg = String()
        msg.data = reference
        self.pub.publish(msg)

    def pub_speed(self, speed):
        speed = self.normalizeSpeed(speed)
        self.speed = speed
        data = {'action': '1', 'speed': float(speed)}
        reference = json.dumps(data)
        msg = String()
        msg.data = reference
        self.pub.publish(msg)