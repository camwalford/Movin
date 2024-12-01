from collections import deque
import numpy as np


default_landmark_weights = np.ones(33)
face_landmark_indices = [8,6,5,4,1,2,3,7,10,9]
default_landmark_weights[face_landmark_indices] = 0.1


class MovementDetector:
    def __init__(self, queue_size=30, threshold=10, z_weight=0.1, landmark_weights=default_landmark_weights):
        self._threshold = threshold
        self._queue_size = queue_size
        self._z_weight = z_weight
        self._landmarks_queue = deque(maxlen=queue_size)

        # Initialize landmark weights
        if landmark_weights is not None:
            self._landmark_weights = np.array(landmark_weights)
            if len(self._landmark_weights) != 33:
                raise ValueError("landmark_weights must have length 33")
        else:
            # Default to uniform weights if none provided
            self._landmark_weights = np.ones(33)

    def movement_detected(self, landmark_array):
        if landmark_array is None:
            print("No landmarks detected in this frame.")
            return False
        self._landmarks_queue.append(landmark_array)
        if self._is_queue_full():
            return self._is_movement_present()
        return False

    def _is_movement_present(self):
        # print(f"Checking difference in landmarks after {self._queue_size} frames.")
        first_frame_landmarks = list(self._landmarks_queue)[:5]
        last_frame_landmarks = list(self._landmarks_queue)[-5:]
        total_movement = self._calculate_distance(first_frame_landmarks, last_frame_landmarks)
        print(f"Total movement: {total_movement}")
        if total_movement > self._threshold:
            print("Movement detected. Resetting queue.")
            self._landmarks_queue.clear()
            return True
        return False

    def _calculate_distance(self, first_landmarks, last_landmarks):
        avg_first = np.mean(first_landmarks, axis=0)
        avg_last = np.mean(last_landmarks, axis=0)
        diffs = avg_first - avg_last
        diffs[:, 2] *= self._z_weight
        distances = np.linalg.norm(diffs, axis=1)
        weighted_distances = distances * self._landmark_weights
        total_distance = np.sum(weighted_distances)
        return total_distance

    def _is_queue_full(self):
        return len(self._landmarks_queue) == self._landmarks_queue.maxlen
