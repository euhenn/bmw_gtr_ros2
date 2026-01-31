# !/usr/bin/python3

import numpy as np
import cv2 as cv
from time import time

LANE_KEEPER_PATH = "models/lane_keeper_small.onnx" #main model for lane keeping
## avg right -0.17356677295197556
DISTANCE_POINT_AHEAD = 0.35

class Detection:

    # init
    def __init__(self) -> None:

        # lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0

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
