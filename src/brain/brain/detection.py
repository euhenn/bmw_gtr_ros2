# !/usr/bin/python3

import numpy as np
import cv2 as cv
import pickle
import collections
from time import time

from stopline import detect_angle

LANE_KEEPER_PATH = "models/lane_keeper_small.onnx" #main model for lane keeping
# LANE_KEEPER_PATH = "models/round_about8001.onnx"
# avg right -0.17356677295197556
DISTANCE_POINT_AHEAD = 0.35
CAR_LENGTH = 0.4

LANE_KEEPER_AHEAD_PATH = "models/lane_keeper_ahead.onnx" # speed challange
DISTANCE_POINT_AHEAD_AHEAD = 0.6

STOPLINE_ESTIMATOR_PATH = "models/stopline_estimator.onnx"
STOPLINE_ESTIMATOR_ADV_PATH = "models/stopline_estimator_advanced.onnx"
PREDICTION_OFFSET = -0.08


class Detection:

    # init
    def __init__(self) -> None:

        # lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0

        # stop line detection advanced
        #self.stopline_estimator_adv = cv.dnn.readNetFromONNX(STOPLINE_ESTIMATOR_ADV_PATH)
        self.est_dist_to_stopline_adv = 1.0
        self.avg_stopline_detection_adv_time = 0
        self.stopline_adv_cnt = 0

    def detect_lane(self, frame, show_ROI=False, faster=False):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2),
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        #frame = cv.blur(frame, (7,7), 0)
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        if faster:
            blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        else:
            frame_flip = cv.flip(frame, 1)
            # stack the 2 images
            images = np.stack((frame, frame_flip), axis=0)
            blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        self.lane_keeper.setInput(blob)
        out = -self.lane_keeper.forward()  # NOTE: MINUS SIGN IF OLD NET
        output = out[0]
        output_flipped = out[1] if not faster else None

        # <++>
        e2 = output[0]
        e3 = output[1]

        if not faster:
            e2_flipped = output_flipped[0]
            e3_flipped = output_flipped[1]

            e2 = (e2 - e2_flipped) / 2
            e3 = (e3 - e3_flipped) / 2

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d+0.2, np.sin(e3)*d])
        # print(f"est_point_ahead: {est_point_ahead}")

        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e2, e3, est_point_ahead


    def detect_stopline(self, frame, show_ROI=True):
        """
        Estimates the distance to the next stop line
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        try:
            # convert to gray
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = frame[int(frame.shape[0]*(2/5)):, :]
            # keep the bottom 2/3 of the image
            frame = cv.blur(frame, (9, 9), 0)
            frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
            frame = cv.Canny(frame, 100, 200)
            frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
            frame = cv.resize(frame, IMG_SIZE)

            blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True,
                                        crop=False)
            self.stopline_estimator_adv.setInput(blob)
            output = self.stopline_estimator_adv.forward()
            stopline_x = dist = output[0][0] + PREDICTION_OFFSET
            stopline_y = output[0][1]
            stopline_angle = output[0][2]
            self.est_dist_to_stopline = dist

            stopline_detection_time = 1000*(time()-start_time)
            self.avg_stopline_detection_time = \
                (self.avg_stopline_detection_time*self.lane_cnt +
                 stopline_detection_time) / (self.lane_cnt+1)
            self.lane_cnt += 1
            if show_ROI:
                cv.imshow('stopline_detection', frame)
                cv.imwrite(f'sd/sd_{int(time()*1000)}.png', frame)
                # cv.waitKey(1)
            #print(f"stopline_detection dist: {dist:.2f}, in {stopline_detection_time:.2f} ms")
            return stopline_x, stopline_y, stopline_angle
        except Exception:
            return 69, 420, 666


    # helper functions
    def automatic_brightness_and_contrast(image, clip_hist_percent=1):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result)

    # into detection class
    def ImageHistogram(kmeans, descriptor_list, no_clusters):
        """
        Compute the histogram of occurrences of the visual words in the
        input image
        """
        im_hist = np.zeros(no_clusters)

        # feature is the descriptor of a single keypoint
        for feature in descriptor_list:

            feature = feature.reshape(1, -1) 
            idx = kmeans.predict(feature)
            im_hist[idx] += 1
        return im_hist

    # into detection class
    def draw_ROI(frame, TL, BR, show_rect=False, prediction=None, conf=None,
                 show_prediction=False):
        # Blue color in BGR
        if show_rect:
            image = frame.copy()
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv.rectangle(image, TL, BR, color=(255, 0, 0), thickness=2)
            cv.imshow("Frame preview", image)
            # cv.waitKey(1)
        if show_rect and show_prediction:
            image = frame.copy()
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv.rectangle(image, TL, BR, color=(255, 0, 0), thickness=2)
            cv.putText(img=image, text=prediction + ' ' + str(conf) + '%',
                       org=(TL[0]-100, TL[1]),
                       fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                       color=(0, 255, 0), thickness=1)
            cv.imshow("Frame preview", image)
            # cv.waitKey(1)

    def detect_yaw_stopline(self, frame, show_ROI=False):
        return detect_angle(original_frame=frame, plot=show_ROI)