from collections import deque

import numpy as np
from tensorflow.python.ops.gen_data_flow_ops import queue_size



class MovementDetector:
    def __init__(self, queue_size=30, threshold=15):
        self._threshold = threshold
        self._queue_size = queue_size
        self._landmarks_queue = deque(maxlen=queue_size)


    def movement_detected(self, landmark_array):
        if landmark_array is None:
            print("No landmarks detected in this frame.")
            return False
        # print("Detecting movement.")
        self._landmarks_queue.append(landmark_array)
        if self._is_queue_full():
            return self._is_movement_present()
        return False

    def _is_movement_present(self):
        print(f"Checking difference in landmarks after {self._queue_size} frames.")
        first_frame_landmarks = self._landmarks_queue[0]
        last_frame_landmarks = self._landmarks_queue[-1]
        total_movement = self._calculate_distance(first_frame_landmarks, last_frame_landmarks)
        print(f"Total movement: {total_movement}")
        if total_movement > self._threshold:
            print("Movement detected. Resetting queue.")
            self._landmarks_queue.clear()
            self._landmarks_queue.append(last_frame_landmarks)
            return True
        return False

    def _calculate_distance(self, first_landmarks, last_landmarks):
        distances = np.linalg.norm(first_landmarks - last_landmarks, axis=1)
        total_distance = np.sum(distances)
        return total_distance

    def _is_queue_full(self):
        return len(self._landmarks_queue) == self._landmarks_queue.maxlen
