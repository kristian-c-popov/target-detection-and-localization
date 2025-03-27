# Single Target Detection and Localization: Classes and Functions

# Last Updated: 03/27/2025

import numpy as np
import matplotlib.pyplot as plt

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

class Camera:
    """A class to represent a calibrated camera with intrinsic and extrinsic parameters."""
    def __init__(self, K, R, t):
        """
        Initializes a camera object.

        Parameters:
            K : (3x3) Intrinsic matrix
            R : (3x3) Rotation matrix
            t : (3x1) Translation vector
        """
        self.K = K
        self.R = R
        self.t = t
        self.P = K @ np.hstack((R, t))  # Projection matrix: P = K[R|t]

def localize(cameras, pixels):
    """
    Triangulates a 3D point using multiple camera views.

    Parameters:
        cameras : List of Camera objects
        pixels  : List of (u, v) pixel coordinates corresponding to each camera

    Returns:
        X : (3x1) 3D coordinates of the point in world frame
    """
    A = []

    for cam, (u, v) in zip(cameras, pixels):
        P = cam.P
        A.append(u * P[2] - P[0])  # Equation for u
        A.append(v * P[2] - P[1])  # Equation for v

    A = np.array(A)

    # Solve Ax = 0 using SVD
    _, _, V = np.linalg.svd(A)
    X_homogeneous = V[-1]
    X = X_homogeneous[:3] / X_homogeneous[3]

    return X

def centroid(bounding_box):
    # Bounding box given in form xyxy
    x01, y01, x02, y02 = bounding_box

    # Find centroid coordinates
    x = (x01 + x02) / 2
    y = (y01 + y02) / 2

    return x, y

def tensor_to_array(boxes_tensor):
    boxes_array = []

    for i in range(len(boxes_tensor)):
        boxes_i = boxes_tensor[i]
        boxes_np = boxes_i.cpu().numpy()
        boxes_1d_array = boxes_np.flatten()
        boxes_array.append(boxes_1d_array)

    return boxes_array
