import cv2
import mediapipe as mp
import os
import numpy as np
import csv
from glob import glob
from datetime import datetime

class BlazePoseVideoProcessor:
    def __init__(self, input_dir='./input', output_dir='./output', min_detection_confidence=0.5):
        """
        Initializes the BlazePoseVideoProcessor.

        Args:
            input_dir (str): Path to the input directory containing 'train' and 'test' subdirectories.
            output_dir (str): Path to the output directory.
            min_detection_confidence (float): Minimum detection confidence for pose estimation.
        """
        # Initialize MediaPipe BlazePose
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_detection_confidence = min_detection_confidence
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence
        )
        # Initialize global frame count
        self.global_frame_count = 0

    def process_all_videos(self):
        # Process videos in both 'train' and 'test' directories
        for dataset_type in ['train', 'test']:
            input_dataset_dir = os.path.join(self.input_dir, dataset_type)
            output_dataset_dir = os.path.join(self.output_dir, dataset_type)

            # Create timestamped subdirectory in the output dataset directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dataset_dir = os.path.join(output_dataset_dir, timestamp)
            frames_output_dir = os.path.join(output_dataset_dir, 'frames')
            labels_csv_path = os.path.join(output_dataset_dir, 'labels.csv')

            # Ensure output directories exist
            os.makedirs(frames_output_dir, exist_ok=True)

            # Initialize the CSV file with headers
            with open(labels_csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['filename', 'labels'])

            # Process videos in the input dataset directory
            video_paths = glob(os.path.join(input_dataset_dir, '*.mp4'))
            if not video_paths:
                print(f"No .mp4 files found in {input_dataset_dir}")
                continue  # Move to the next dataset_type

            print(f"Processing '{dataset_type}' videos...")

            for video_path in video_paths:
                print(f"Processing video: {video_path}")
                self.process_video(
                    video_path=video_path,
                    frames_output_dir=frames_output_dir,
                    labels_csv_path=labels_csv_path
                )

        self.pose.close()
        print("Processed all videos.")

    def process_video(self, video_path, frames_output_dir, labels_csv_path):
        # Process a single video
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Finished processing video: {video_path}")
                break

            self.global_frame_count += 1
            processed_frame = self._process_frame(frame, video_name, frames_output_dir, labels_csv_path)

            # Display the frame with landmarks (optional)
            # cv2.imshow('BlazePose Processed Video', processed_frame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

        cap.release()
        # cv2.destroyAllWindows()  # Commented out since display is optional

    def _process_frame(self, frame, video_name, frames_output_dir, labels_csv_path):
        # Convert the frame to RGB as MediaPipe BlazePose expects RGB input
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Make a copy of the original frame before drawing landmarks
        original_frame = frame.copy()

        if results.pose_landmarks:
            self._draw_landmarks(frame, results.pose_landmarks)
            labels = self._calculate_angles_and_get_labels(results.pose_landmarks, w, h)
            # Save the original frame and labels
            self._save_frame_and_labels(
                frame=original_frame,
                labels=labels,
                video_name=video_name,
                frames_output_dir=frames_output_dir,
                labels_csv_path=labels_csv_path
            )
        else:
            # If no pose is detected, you may choose to label accordingly or skip the frame
            pass

        return frame  # This frame has landmarks drawn on it

    def _draw_landmarks(self, frame, pose_landmarks):
        # Draw pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    def _calculate_angles_and_get_labels(self, pose_landmarks, w, h):
        # Extract coordinates of key landmarks for angle calculation
        def get_coords(landmark):
            return np.array([landmark.x * w, landmark.y * h])

        # Initialize list of contracted muscles
        contracted_muscles = []

        # Left Side Angles
        left_shoulder = get_coords(pose_landmarks.landmark[11])
        left_elbow = get_coords(pose_landmarks.landmark[13])
        left_wrist = get_coords(pose_landmarks.landmark[15])
        left_hip = get_coords(pose_landmarks.landmark[23])

        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        left_shoulder_angle = self._calculate_angle(left_hip, left_shoulder, left_elbow)

        # Determine contracted muscles based on angles
        # Left Biceps
        if left_elbow_angle < 130:
            contracted_muscles.append('left_biceps')
        # Left Triceps
        else:
            contracted_muscles.append('left_triceps')

        # # Left Deltoid
        # if left_shoulder_angle < 90:
        #     contracted_muscles.append('left_deltoid')

        # Right Side Angles
        right_shoulder = get_coords(pose_landmarks.landmark[12])
        right_elbow = get_coords(pose_landmarks.landmark[14])
        right_wrist = get_coords(pose_landmarks.landmark[16])
        right_hip = get_coords(pose_landmarks.landmark[24])

        right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        right_shoulder_angle = self._calculate_angle(right_hip, right_shoulder, right_elbow)

        # Right Biceps
        if right_elbow_angle < 130:
            contracted_muscles.append('right_biceps')
        # Right Triceps
        else:
            contracted_muscles.append('right_triceps')

        # # Right Deltoid
        # if right_shoulder_angle < 90:
        #     contracted_muscles.append('right_deltoid')

        # Return the list of contracted muscles
        labels = ','.join(contracted_muscles)

        return {
            'labels': labels
        }

    def _calculate_angle(self, pointA, pointB, pointC):
        # Calculate the angle at pointB using 2D coordinates
        BA = pointA - pointB
        BC = pointC - pointB
        cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure within valid range
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _save_frame_and_labels(self, frame, labels, video_name, frames_output_dir, labels_csv_path):
        # Save the original frame image (without landmarks)
        frame_filename = f'{video_name}_frame_{self.global_frame_count:06d}.jpg'
        frame_path = os.path.join(frames_output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Append the label data to the CSV file
        with open(labels_csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                frame_filename,
                labels['labels']
            ])

    def _display_angle(self, frame, joint_coords, angle_text):
        # Display the angle text near the joint on the frame
        x, y = int(joint_coords[0]), int(joint_coords[1])
        cv2.putText(frame, angle_text, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


if __name__ == '__main__':
    # Initialize the BlazePoseVideoProcessor
    processor = BlazePoseVideoProcessor(input_dir='labeller_input',
                                        eroutput_dir='classifier_input')
    processor.process_all_videos()
