# Single Target Detection and Localization -- Active Tracking Cameras

# Last Updated: 03/27/2025

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from single_target_dnl import *

# Interact with User as program starts
print("\n")
print("SINGLE TARGET DETECTION AND LOCALIZATION -- ACTIVE TRACKING CAMERAS")

# Load model
model = YOLO('MODEL_NAME.pt')

# Relative Camera Positions
C1 = np.array([[0], [0], [0]])
C2 = np.array([[3], [0], [0]])

# Intrinsic matrix for cameras -- EXAMPLE:
K = np.array([[1000, 0, 320],
              [0, 1000, 240], 
              [0, 0, 1]])

K1 = K
K2 = K

# Camera: Either use camera index (0,1,2,...,n) or 'http:// xxx_IP_xxx :8080/video'
cf1 = 0
cf2 = 1

# Open camera feeds
caps = [cv2.VideoCapture(feed) for feed in [cf1, cf2]]

# Open CSV file for writing
with open("output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["X", "Y", "Z"]) # Header row

    # plot() Not implemented in public.03.27.2025 Revision
    #points_history = []

    while True:
        # Read in camera rotation matrics from respective .csv
        R1 = np.loadtxt('c1_rotation.csv', delimiter=',')
        R2 = np.loadtxt('c2_rotation.csv', delimiter=',')
        #RX = np.loadtxt('cX_rotation.csv', delimiter=',')

        # Translation Vectors
        t1 = -R1 @ C1
        t2 = -R2 @ C2

        # Define camera objects with intrinsic and extrinsic parameters
        cameras = [
            Camera(K1, R1, t1),
            Camera(K2, R2, t2),
        ]

        frames = []
        valid_frames = True
        
        # Capture frames from all cameras
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                valid_frames = False
                break
            frames.append(frame)
        
        if not valid_frames:
            print("Failed to read from one or more cameras.")
            break
        
        # Run model on all frames
        results = [model.predict(source=frame, conf=0.2) for frame in frames]
        
        # Ensure all cameras have detections
        if all(len(result[0].boxes) > 0 for result in results):
            # Convert xyxy tensor to NumPy array
            boxes = [tensor_to_array(result) for result in results]
            
            # Extract single target centroids from each camera
            centroids = [centroid(box[0]) for box in boxes]

            # If you have issues with STDnL, add call to match_detects here
            # Calling match_detects increases inference time, so if you can
            #  get away with not calling it, don't call it
            
            # Localize the 3D point
            X = localize(cameras, centroids)
            writer.writerow(X)
            # plot() Not implemented in public.03.27.2025 Revision
            #points_history.append(tuple(X))
    
        # plot() Not implemented in public.03.27.2025 Revision
        #plot(points_history)
    
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release all cameras and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()

