import cv2
import mediapipe as mp
import numpy as np
import os
import keras
from collections import deque

class Labeller:

    def __init__(self):
        print("Setting up Blaze.")
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                             model_complexity=1,
                             smooth_landmarks=True,
                             enable_segmentation=False,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)


    def extract_landmarks(self, image):
        print("Labeller received image:", image)
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return np.zeros(99)  # Return a zero array if no landmarks are detected

        print("Extracting landmarks.")
        landmarks = results.pose_landmarks.landmark
        pose = []
        for landmark in landmarks:
            pose.extend([landmark.x, landmark.y, landmark.z])

        print("Returning landmarks.")
        return np.array(pose)
