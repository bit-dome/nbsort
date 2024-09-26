import cv2

import numpy as np
import copy

class GMC:
    def __init__(self, downscale=1, verbose=None):
        super(GMC, self).__init__()
        self.downscale = max(1, int(downscale))

        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                   useHarrisDetector=False, k=0.04)

        self.prev_frame = None
        self.prev_kp = None
        self.init_frame = False

    def apply(self, raw_frame, detections=None):
        h, w, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        if self.downscale > 1:
            frame = cv2.resize(frame, (w // self.downscale, h // self.downscale))

        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        if not self.init_frame:
            self.prev_frame = frame.copy()
            self.prev_kp = copy.copy(keypoints)
            self.init_frame = True
            return H

        matched_kp, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, self.prev_kp, None)

        prev_pts = [self.prev_kp[i] for i in range(len(status)) if status[i]]
        curr_pts = [matched_kp[i] for i in range(len(status)) if status[i]]

        prev_pts = np.array(prev_pts)
        curr_pts = np.array(curr_pts)

        if len(prev_pts) > 4:
            H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
            if self.downscale > 1:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        self.prev_frame = frame.copy()
        self.prev_kp = copy.copy(keypoints)

        return H
