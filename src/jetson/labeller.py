import mediapipe as mp
import numpy as np


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
        # print("Extracting landmarks.")

        results = self.pose.process(image)
        if not results.pose_landmarks:
            return None, None

        landmarks = results.pose_landmarks.landmark
        pose = []
        for landmark in landmarks:
            pose.append([landmark.x, landmark.y, landmark.z])

        non_flattened_landmarks = np.array(pose)  # Shape: (33, 3)
        flattened_landmarks = non_flattened_landmarks.flatten()  # Shape: (99,)

        # print(f"Shape of flattened_landmarks: {flattened_landmarks.shape}")
        # print(f"Shape of non_flattened_landmarks: {non_flattened_landmarks.shape}")
        return flattened_landmarks, non_flattened_landmarks
