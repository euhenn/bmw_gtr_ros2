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
import time as system_time
import helper_functions as hf
from automobile_data_interface import Automobile_Data
from ukf_intf import CarUKF

REALISTIC = False                   # Set this to True when using realistic servo model
GAZEBO_WITH_STEP_CONTROL = False    # Set this to True when using step control

ENCODER_TIMER = 0.01  # frequency of encoder reading
STEER_UPDATE_FREQ = 50.0 if REALISTIC else 150.0
SERVO_DEAD_TIME_DELAY = 0.15 if REALISTIC else 0.0
MAX_SERVO_ANGULAR_VELOCITY = 2.8 if REALISTIC else 15.0
DELTA_ANGLE = np.rad2deg(MAX_SERVO_ANGULAR_VELOCITY) / STEER_UPDATE_FREQ
MAX_STEER_COMMAND_FREQ = 50.0
MAX_STEER_SAMPLES = max(int((2 * SERVO_DEAD_TIME_DELAY) * MAX_STEER_COMMAND_FREQ), 10)

GPS_DELAY = 0.45            # [s] delay for gps message to arrive
ENCODER_POS_FREQ = 100.0    # [Hz] frequency of encoder position messages
GPS_FREQ = 10.0             # [Hz] frequency of gps messages
BUFFER_PAST_MEASUREMENTS_LENGTH = int(round(GPS_DELAY * ENCODER_POS_FREQ))


class AutomobileDataSimulator(Node, Automobile_Data):
    """ROS2 node that simulates and publishes automobile data."""

    def __init__(self,
                 trig_control=True,
                 trig_bno=False,
                 trig_enc=False,
                 trig_cam=False,
                 trig_gps=False):
        Node.__init__(self, 'automobile_data_simulator')
        Automobile_Data.__init__(self)

        self.timestamp = 0.0
        self.prev_x_true = self.x_true
        self.prev_y_true = self.y_true
        self.prev_timestamp = 0.0
        self.velocity_buffer = collections.deque(maxlen=20)
        self.target_steer = 0.0
        self.curr_steer = 0.0
        self.steer_deque = collections.deque(maxlen=MAX_STEER_SAMPLES)
        
        if GAZEBO_WITH_STEP_CONTROL:
            # using ROS simulation time for step control mode
            self.time_last_steer_command = self.get_clock().now()
        else:
            self.time_last_steer_command = system_time.time()
            
        self.target_dist = 0.0
        self.arrived_at_dist = True
        self.yaw_true = 0.0

        self.x_buffer = collections.deque(maxlen=5)
        self.y_buffer = collections.deque(maxlen=5)

        self.ukf_filter = CarUKF(wheelbase=self.WB)

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

    def get_current_time(self):
        """Get current time based on the selected mode."""
        if GAZEBO_WITH_STEP_CONTROL:
            return self.get_clock().now()
        else:
            return system_time.time()

    def get_time_difference(self, t1, t2):
        """Calculate time difference based on the selected mode."""
        if GAZEBO_WITH_STEP_CONTROL:
            # Here both are ROS Time objects
            return (t2 - t1).nanoseconds / 1e9
        else:
            # And here time floats
            return t2 - t1

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
        self.x_gps = self.x
        self.y_gps = self.y
        self.ukf_filter.update_sensor('gps', (data.pos_x, data.pos_y))
        self.x_est, self.y_est = self.ukf_filter.state[:2]

    def imu_callback(self, data):
        self.roll = float(data.roll)
        self.roll_deg = np.rad2deg(self.roll)
        self.pitch = float(data.pitch)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_true = float(data.yaw)                 # Comes wraped to [-π, π]
        self.yaw = float(data.yaw) + self.yaw_offset
        self.yaw_deg = np.rad2deg(self.yaw)
        true_posL = np.array([data.posx, data.posy])
        true_posR = hf.mL2mR(true_posL)
        self.x_true = true_posR[0] - self.WB / 2 * np.cos(self.yaw_true)
        self.y_true = true_posR[1] - self.WB / 2 * np.sin(self.yaw_true)
        self.timestamp = float(data.timestamp)
        
        self.ukf_filter.update_sensor('imu', float(data.yaw))
        self.yaw_est = self.ukf_filter.state[2]

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
            curr_time = self.get_current_time()
            angle, t = self.steer_deque.popleft()
            
            # Calculate time difference based on mode
            time_diff = self.get_time_difference(t, curr_time)
            
            if time_diff < SERVO_DEAD_TIME_DELAY:
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
        v = float(data)
        self.ukf_filter.predict(v_meas=v, steer_angle=self.curr_steer)
        self.ukf_filter.update_sensor('encoder', v)
        self.x_est, self.y_est, self.yaw_est, _ = self.ukf_filter.state

    # === COMMAND ACTIONS ===
    def drive_speed(self, speed=0.0):
        self.arrived_at_dist = True
        self.pub_speed(speed)

    def drive_angle(self, angle=0.0, direct=False):
        angle = self.normalizeSteer(angle)
        curr_time = self.get_current_time()
        
        # Calculate time difference based on mode
        time_diff = self.get_time_difference(self.time_last_steer_command, curr_time)
        
        if time_diff > 1 / MAX_STEER_COMMAND_FREQ:
            self.time_last_steer_command = curr_time
            self.steer_deque.append((angle, curr_time))
        #else:
        #    self.get_logger().warn('Missed steer command...')

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
        curr_time = self.get_current_time()
        self.steer_deque.append((angle, curr_time))
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