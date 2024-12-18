import cv2
import mediapipe as mp
import os
import numpy as np
from glob import glob
from datetime import datetime
import yaml
from tqdm import tqdm
from mediapipe.python.solutions.pose import PoseLandmark
import pandas as pd
from src.utils.custom_logger import setup_logger
import logging


class BlazePoseVideoLabeller:
    def __init__(self, config):
        """
        Initializes the BlazePoseVideoLabeller.

        Args:
            config (dict): Configuration dictionary.
        """
        self.logger = setup_logger(
            name="BlazePoseVideoProcessor",
            log_dir=config.get('log_dir', 'logs'),
            log_level=config.get('log_level', 'INFO')
        )
        self.logger.info("Initializing BlazePoseVideoProcessor...")

        self.input_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.debug(f"Configuration: {config}")

        # Initialize MediaPipe BlazePose with settings
        try:
            self.pose = mp.solutions.pose.Pose(**config['blaze_settings'])
            self.logger.info("MediaPipe BlazePose initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe BlazePose: {e}")
            raise e

        self.labels = config['movement_labels']
        self.multi_label = config.get('multi_label', False)
        self.target_label = config.get('target_label', None)
        self.use_joint_angles = config.get('use_joint_angles', False)
        self.save_overlay_video = config.get('save_overlay_video', False)
        self.global_frame_count = 0  # Counter for frames processed
        self.logger.info("BlazePoseVideoProcessor initialized successfully.")

    def process_all_videos(self):
        """
        Processes all labeled videos in the input directory and saves outputs to Parquet files.
        """
        try:
            self._adjust_labels_based_on_config()

            for label_name, label_data in self.labels.items():
                joint_angles = label_data.get('joints', [])
                self.logger.info(f"Processing label: {label_name} with {len(joint_angles)} joint angles.")

                for dataset_type in ['train', 'test']:
                    self._process_dataset_type(label_name, joint_angles, dataset_type)

            self.pose.close()
            self.logger.info("Processed all videos successfully.")
        except Exception as e:
            self.logger.exception(f"An error occurred while processing videos: {e}")

    def _adjust_labels_based_on_config(self):
        """
        Adjusts self.labels based on whether multi_label is disabled and a target_label is specified.
        """
        if not self.multi_label and self.target_label:
            self.labels = {self.target_label: self.labels.get(self.target_label, {})}
            self.logger.info(f"Multi-label disabled. Target label set to: {self.target_label}")
        elif not self.multi_label and not self.target_label:
            self.logger.warning("Multi-label is disabled but no target_label is specified.")

    def _process_dataset_type(self, label_name, joint_angles, dataset_type):
        """
        Processes a specific dataset type (train or test) for a given label.

        Args:
            label_name (str): The label name.
            joint_angles (list): List of joint angle configurations.
            dataset_type (str): 'train' or 'test'.
        """
        input_dataset_dir, output_dataset_dir, frames_output_dir, labels_parquet_path, overlay_video_path = \
            self._setup_output_directories(label_name, dataset_type)

        labels_data = []
        video_paths = glob(os.path.join(input_dataset_dir, '*.mp4'))

        if not video_paths:
            self.logger.warning(f"No .mp4 files found in {input_dataset_dir}")
            return

        self.logger.info(f"Processing '{dataset_type}' videos for label '{label_name}'...")
        for video_path in tqdm(video_paths, desc=f"Videos ({dataset_type}-{label_name})", unit='video'):
            self.logger.info(f"Processing video: {video_path}")
            self.process_video(
                video_path=video_path,
                frames_output_dir=frames_output_dir,
                labels_data=labels_data,
                label_name=label_name,
                joint_angles=joint_angles,
                overlay_video_path=overlay_video_path
            )

        self._save_labels_data_to_parquet(labels_data, labels_parquet_path, label_name, dataset_type)

    def _setup_output_directories(self, label_name, dataset_type):
        """
        Sets up the output directories for frames, labels, and overlay videos.

        Args:
            label_name (str): Label name.
            dataset_type (str): 'train' or 'test'.

        Returns:
            tuple: (input_dataset_dir, output_dataset_dir, frames_output_dir, labels_parquet_path, overlay_video_path)
        """
        input_dataset_dir = os.path.join(self.input_dir, dataset_type, label_name)
        output_dataset_dir = os.path.join(self.output_dir, dataset_type, label_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dataset_dir = os.path.join(output_dataset_dir, timestamp)
        frames_output_dir = os.path.join(output_dataset_dir, 'frames')
        labels_parquet_path = os.path.join(output_dataset_dir, 'labels.parquet')
        overlay_video_path = os.path.join(output_dataset_dir, f'overlay_{label_name}')

        os.makedirs(frames_output_dir, exist_ok=True)
        self.logger.debug(f"Created output directories at {output_dataset_dir}")

        return input_dataset_dir, output_dataset_dir, frames_output_dir, labels_parquet_path, overlay_video_path

    def _save_labels_data_to_parquet(self, labels_data, labels_parquet_path, label_name, dataset_type):
        """
        Saves the collected label data to a Parquet file.

        Args:
            labels_data (list): Collected labels data.
            labels_parquet_path (str): Path to save the parquet file.
            label_name (str): Label name.
            dataset_type (str): 'train' or 'test'.
        """
        if labels_data:
            df_labels = pd.DataFrame(labels_data)
            try:
                df_labels.to_parquet(labels_parquet_path, index=False)
                self.logger.info(f"Saved labels to {labels_parquet_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Parquet file {labels_parquet_path}: {e}")
        else:
            self.logger.warning(f"No labels data collected for {label_name} in {dataset_type}.")

    def process_video(self, video_path, frames_output_dir, labels_data, label_name, joint_angles, overlay_video_path):
        """
        Processes a single video to extract frames, landmarks, and optionally joint angles.
        Saves the output as frames and updates labels_data. Optionally saves an overlay video.

        Args:
            video_path (str): Path to the input video.
            frames_output_dir (str): Directory to save extracted frames and landmarks.
            labels_data (list): List to append label data dictionaries.
            label_name (str): Label associated with the video.
            joint_angles (list): List of joint angle configurations.
            overlay_video_path (str): Path to save the overlay video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0] # Get video name without extension
        frame_width, frame_height, fps, total_frames = self._get_video_properties(cap)

        overlay_writer = self._initialize_overlay_writer(overlay_video_path, video_name, frame_width, frame_height, fps)

        frame_index = 0
        try:
            with tqdm(total=total_frames, desc=f"Frames ({video_name})", unit='frame') as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.debug(f"End of video reached: {video_path}")
                        break  # End of video

                    frame_index += 1
                    self.global_frame_count += 1
                    self.logger.debug(f"Processing frame {frame_index}/{total_frames} of video '{video_name}'.")

                    processed_frame = self._process_frame(
                        frame=frame,
                        video_name=video_name,
                        frames_output_dir=frames_output_dir,
                        labels_data=labels_data,
                        label_name=label_name,
                        joint_angles=joint_angles
                    )

                    if self.save_overlay_video and overlay_writer is not None:
                        overlay_writer.write(processed_frame)
                        self.logger.debug(f"Wrote overlay frame {frame_index} to overlay video")

                    pbar.update(1)
        except Exception as e:
            self.logger.exception(f"An error occurred while processing video '{video_path}': {e}")
        finally:
            cap.release()
            self.logger.debug(f"Released video capture for {video_path}")
            if overlay_writer is not None:
                overlay_writer.release()
                self.logger.debug(f"Released overlay video writer.")

    def _get_video_properties(self, cap):
        """
        Retrieves video properties such as width, height, fps, and total frames.

        Args:
            cap (cv2.VideoCapture): Opened video capture object.

        Returns:
            tuple: (frame_width, frame_height, fps, total_frames)
        """
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return frame_width, frame_height, fps, total_frames

    def _initialize_overlay_writer(self, overlay_video_path, video_name, frame_width, frame_height, fps):
        """
        Initializes the overlay video writer if saving overlay video is enabled.

        Args:
            overlay_video_path (str): Base path for the overlay video.
            video_name (str): Name of the video file.
            frame_width (int): Video frame width.
            frame_height (int): Video frame height.
            fps (float): Frames per second of the video.

        Returns:
            cv2.VideoWriter or None: The initialized video writer or None if disabled or failed.
        """
        if not self.save_overlay_video:
            return None

        try:
            overlay_video_path = f"{overlay_video_path}_{video_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (frame_width, frame_height))
            self.logger.debug(f"Overlay video writer initialized at {overlay_video_path}")
            return overlay_writer
        except Exception as e:
            self.logger.error(f"Failed to initialize overlay video writer: {e}")
            return None

    def _process_frame(self, frame, video_name, frames_output_dir, labels_data, label_name, joint_angles):
        """
        Processes a single frame: detects pose, draws landmarks, calculates joint angles,
        overlays angles, and saves data.

        Args:
            frame (ndarray): The video frame.
            video_name (str): Name of the video file.
            frames_output_dir (str): Directory to save frames and landmarks.
            labels_data (list): List to append label data dictionaries.
            label_name (str): Label associated with the video.
            joint_angles (list): List of joint angle configurations.

        Returns:
            ndarray: The frame with overlays drawn (if any).
        """
        try:
            results = self._detect_pose(frame)
            overlay_frame = frame.copy()

            if results.pose_landmarks:
                self._draw_landmarks(overlay_frame, results.pose_landmarks)
                angles_dict = self._calculate_joint_angles(
                    results.pose_landmarks, joint_angles, overlay_frame.shape[1], overlay_frame.shape[0]
                )
                self._overlay_joint_angles(overlay_frame, results.pose_landmarks, angles_dict,
                                           overlay_frame.shape[1], overlay_frame.shape[0])

                angles_match = self._evaluate_angles(angles_dict, joint_angles)
                self._display_label(overlay_frame, label_name, match=angles_match)

                if angles_match:
                    self._save_frame_and_labels(
                        frame=frame,
                        pose_landmarks=results.pose_landmarks,
                        pose_world_landmarks=results.pose_world_landmarks,
                        label_name=label_name,
                        video_name=video_name,
                        frames_output_dir=frames_output_dir,
                        labels_data=labels_data,
                        joint_angles=angles_dict
                    )
                    self.logger.info(f"Frame {self.global_frame_count} matched and saved.")
                else:
                    self.logger.debug(f"Frame {self.global_frame_count} did not match label '{label_name}'.")
            else:
                self.logger.debug(f"No pose detected in frame {self.global_frame_count} of video '{video_name}'.")

            return overlay_frame
        except Exception as e:
            self.logger.exception(f"An error occurred while processing frame {self.global_frame_count}: {e}")
            return frame  # Return the original frame if processing fails

    def _detect_pose(self, frame):
        """
        Detects the pose landmarks from a given frame using MediaPipe Pose.

        Args:
            frame (ndarray): The video frame in BGR format.

        Returns:
            results: The results object from MediaPipe pose processing.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        self.logger.debug("Pose processing completed for current frame.")
        return results

    def _draw_landmarks(self, frame, pose_landmarks):
        """
        Draws pose landmarks on the frame.

        Args:
            frame (ndarray): The video frame.
            pose_landmarks (LandmarkList): Detected pose landmarks.
        """
        try:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                )
            )
            self.logger.debug("Pose landmarks successfully drawn.")
        except Exception as e:
            self.logger.error(f"Failed to draw landmarks: {e}")

    def _display_label(self, frame, label_name, match):
        """
        Displays label and match status on the frame.

        Args:
            frame (ndarray): The video frame.
            label_name (str): The label name.
            match (bool): Whether the frame matches the label.
        """
        try:
            text = f"Label: {label_name} - {'Matched' if match else 'Not Matched'}"
            color = (0, 255, 0) if match else (0, 0, 255)
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )
            self.logger.debug(f"Displayed label on frame: {text}")
        except Exception as e:
            self.logger.error(f"Failed to display label on frame: {e}")

    def _overlay_joint_angles(self, frame, pose_landmarks, angles_dict, frame_width, frame_height):
        """
        Overlays joint angles on the frame near the corresponding joints.

        Args:
            frame (ndarray): The video frame.
            pose_landmarks (LandmarkList): Detected pose landmarks.
            angles_dict (dict): Dictionary of calculated joint angles.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.
        """
        try:
            for joint_name, angle in angles_dict.items():
                self._overlay_single_angle(frame, pose_landmarks, joint_name, angle, frame_width, frame_height)
        except Exception as e:
            self.logger.error(f"Failed to overlay joint angles: {e}")

    def _overlay_single_angle(self, frame, pose_landmarks, joint_name, angle, frame_width, frame_height):
        """
        Overlays a single joint angle near the corresponding joint.

        Args:
            frame (ndarray): The video frame.
            pose_landmarks (LandmarkList): Detected pose landmarks.
            joint_name (str): The joint name key (e.g., 'KNEE_LEFT').
            angle (float): The joint angle in degrees.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.
        """
        try:
            parts = joint_name.split('_')
            name = parts[0].upper()  # Joint name (e.g., KNEE)
            side = parts[1].upper() if len(parts) > 1 else 'BOTH'

            landmark_name = self._get_landmark_for_joint(name, side)
            if landmark_name:
                coords = self._get_coords(pose_landmarks, landmark_name, frame_width, frame_height)
                if coords is not None:
                    x, y = int(coords[0]), int(coords[1])
                    angle_text = f"{name}: {int(angle)}°"
                    offset_x, offset_y = 20, -20
                    cv2.putText(
                        frame,
                        angle_text,
                        (x + offset_x, y + offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    self.logger.debug(f"Overlaid angle '{angle_text}' at ({x + offset_x}, {y + offset_y}).")
        except Exception as e:
            self.logger.error(f"Failed to overlay single angle '{joint_name}': {e}")

    def _get_landmark_for_joint(self, joint_name, side):
        """
        Maps a joint and side to a MediaPipe landmark name.

        Args:
            joint_name (str): Joint name in uppercase (e.g., 'KNEE', 'ELBOW').
            side (str): Side string ('LEFT', 'RIGHT', or 'BOTH').

        Returns:
            str or None: Corresponding MediaPipe landmark name or None if not found.
        """
        if joint_name == "KNEE":
            return 'RIGHT_KNEE' if 'RIGHT' in side else 'LEFT_KNEE'
        elif joint_name == "ELBOW":
            return 'RIGHT_ELBOW' if 'RIGHT' in side else 'LEFT_ELBOW'
        elif joint_name == "SHOULDER":
            return 'RIGHT_SHOULDER' if 'RIGHT' in side else 'LEFT_SHOULDER'
        elif joint_name == "HIP":
            return 'RIGHT_HIP' if 'RIGHT' in side else 'LEFT_HIP'
        else:
            return 'NOSE'

    def _calculate_joint_angles(self, pose_landmarks, joint_angles, frame_width, frame_height):
        """
        Calculates joint angles specified in joint_angles configuration.

        Args:
            pose_landmarks (LandmarkList): Detected pose landmarks.
            joint_angles (list): List of joint angle configurations.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.

        Returns:
            dict: Dictionary of calculated joint angles.
        """
        angles_dict = {}
        try:
            for joint in joint_angles:
                name = joint.get('name')
                side = joint.get('side', 'BOTH').upper()
                self.logger.debug(f"Calculating angle for joint '{name}' on side '{side}'.")
                angle_left, angle_right = self._calculate_joint_angle_by_name(name, pose_landmarks, frame_width, frame_height)

                # Store angles based on side
                if side == 'LEFT':
                    if angle_left is not None:
                        angles_dict[f"{name}_LEFT"] = angle_left
                elif side == 'RIGHT':
                    if angle_right is not None:
                        angles_dict[f"{name}_RIGHT"] = angle_right
                elif side == 'BOTH':
                    if angle_left is not None:
                        angles_dict[f"{name}_LEFT"] = angle_left
                    if angle_right is not None:
                        angles_dict[f"{name}_RIGHT"] = angle_right
                else:
                    self.logger.warning(f"Invalid side '{side}' specified for joint '{name}'.")

            return angles_dict
        except Exception as e:
            self.logger.exception(f"An error occurred while calculating joint angles: {e}")
            return {}

    def _calculate_joint_angle_by_name(self, name, pose_landmarks, frame_width, frame_height):
        """
        Calculates joint angles for a specific joint name (e.g., KNEE, ELBOW, SHOULDER, HIP).

        Args:
            name (str): Joint name (KNEE, ELBOW, SHOULDER, HIP).
            pose_landmarks (LandmarkList): Detected pose landmarks.
            frame_width (int): Video frame width.
            frame_height (int): Video frame height.

        Returns:
            tuple: (angle_left, angle_right) in degrees or (None, None) if calculation fails.
        """
        angle_left, angle_right = None, None
        name = name.upper()

        if name == "KNEE":
            angle_left = self._calculate_angle(
                self._get_coords(pose_landmarks, 'LEFT_HIP', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_KNEE', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_ANKLE', frame_width, frame_height)
            )
            angle_right = self._calculate_angle(
                self._get_coords(pose_landmarks, 'RIGHT_HIP', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_KNEE', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_ANKLE', frame_width, frame_height)
            )
        elif name == "ELBOW":
            angle_left = self._calculate_angle(
                self._get_coords(pose_landmarks, 'LEFT_SHOULDER', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_ELBOW', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_WRIST', frame_width, frame_height)
            )
            angle_right = self._calculate_angle(
                self._get_coords(pose_landmarks, 'RIGHT_SHOULDER', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_ELBOW', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_WRIST', frame_width, frame_height)
            )
        elif name == "SHOULDER":
            angle_left = self._calculate_angle(
                self._get_coords(pose_landmarks, 'LEFT_ELBOW', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_SHOULDER', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_HIP', frame_width, frame_height)
            )
            angle_right = self._calculate_angle(
                self._get_coords(pose_landmarks, 'RIGHT_ELBOW', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_SHOULDER', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_HIP', frame_width, frame_height)
            )
        elif name == "HIP":
            angle_left = self._calculate_angle(
                self._get_coords(pose_landmarks, 'LEFT_SHOULDER', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_HIP', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'LEFT_KNEE', frame_width, frame_height)
            )
            angle_right = self._calculate_angle(
                self._get_coords(pose_landmarks, 'RIGHT_SHOULDER', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_HIP', frame_width, frame_height),
                self._get_coords(pose_landmarks, 'RIGHT_KNEE', frame_width, frame_height)
            )
        else:
            self.logger.warning(f"Unknown or unsupported joint name: '{name}'. Skipping this joint.")
        return angle_left, angle_right

    def _evaluate_angles(self, angles_dict, joint_angles):
        """
        Evaluates whether the calculated joint angles match the configured thresholds if use_joint_angles is True.

        Args:
            angles_dict (dict): Dictionary of calculated joint angles.
            joint_angles (list): List of joint angle configurations.

        Returns:
            bool: True if angles match thresholds or if angle checking is disabled, False otherwise.
        """
        if not self.use_joint_angles:
            return True
        return self._check_joint_angle_thresholds(angles_dict, joint_angles)

    def _check_joint_angle_thresholds(self, angles_dict, joint_angles):
        """
        Checks if the calculated joint angles match the configured thresholds.

        Args:
            angles_dict (dict): Dictionary of calculated joint angles.
            joint_angles (list): List of joint angle configurations.

        Returns:
            bool: True if all joint angles match thresholds, False otherwise.
        """
        try:
            for joint in joint_angles:
                name = joint.get('name')
                side = joint.get('side', 'BOTH').upper()
                min_angle = joint.get('min_angle')
                max_angle = joint.get('max_angle')
                self.logger.debug(
                    f"Checking angle thresholds for joint '{name}' on side '{side}' "
                    f"with thresholds [{min_angle}, {max_angle}]."
                )

                if not self._check_single_joint_threshold(angles_dict, name, side, min_angle, max_angle):
                    return False

            self.logger.debug("All joint angles matched the thresholds.")
            return True
        except Exception as e:
            self.logger.exception(f"An error occurred while checking joint angle thresholds: {e}")
            return False

    def _check_single_joint_threshold(self, angles_dict, name, side, min_angle, max_angle):
        """
        Checks a single joint's angle threshold.

        Args:
            angles_dict (dict): Dictionary of calculated joint angles.
            name (str): Joint name.
            side (str): 'LEFT', 'RIGHT', or 'BOTH'.
            min_angle (float): Minimum acceptable angle.
            max_angle (float): Maximum acceptable angle.

        Returns:
            bool: True if within threshold, False otherwise.
        """
        if side == 'LEFT':
            angle = angles_dict.get(f"{name}_LEFT")
            if angle is None or not (min_angle <= angle <= max_angle):
                self.logger.debug(f"Left {name} angle {angle} out of thresholds.")
                return False
        elif side == 'RIGHT':
            angle = angles_dict.get(f"{name}_RIGHT")
            if angle is None or not (min_angle <= angle <= max_angle):
                self.logger.debug(f"Right {name} angle {angle} out of thresholds.")
                return False
        elif side == 'BOTH':
            angle_left = angles_dict.get(f"{name}_LEFT")
            angle_right = angles_dict.get(f"{name}_RIGHT")
            if (angle_left is None or not (min_angle <= angle_left <= max_angle) or
                    angle_right is None or not (min_angle <= angle_right <= max_angle)):
                self.logger.debug(f"One or both {name} angles out of thresholds.")
                return False
        else:
            self.logger.warning(f"Invalid side '{side}' specified for joint '{name}'.")
            return False
        return True

    def _get_coords(self, pose_landmarks, landmark_name, frame_width, frame_height):
        """
        Retrieves the pixel coordinates for a given landmark.

        Args:
            pose_landmarks (LandmarkList): Detected pose landmarks.
            landmark_name (str): Name of the landmark in PoseLandmark enum.
            frame_width (int): Frame width for scaling.
            frame_height (int): Frame height for scaling.

        Returns:
            np.ndarray or None: Coordinates [x, y] or None if not found.
        """
        try:
            landmark = pose_landmarks.landmark[PoseLandmark[landmark_name.upper()].value]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            coords = np.array([x, y])
            self.logger.debug(f"Retrieved coordinates for {landmark_name}: {coords}")
            return coords
        except KeyError:
            self.logger.error(f"Landmark '{landmark_name}' not found in PoseLandmark.")
            return None
        except Exception as e:
            self.logger.exception(f"An error occurred while getting coordinates for '{landmark_name}': {e}")
            return None

    def _calculate_angle(self, pointA, pointB, pointC):
        """
        Calculates the angle at pointB formed by the line segments BA and BC.

        Args:
            pointA (np.ndarray): Coordinates of point A.
            pointB (np.ndarray): Coordinates of point B.
            pointC (np.ndarray): Coordinates of point C.

        Returns:
            float or None: Angle in degrees or None if calculation fails.
        """
        try:
            if pointA is None or pointB is None or pointC is None:
                self.logger.warning("One or more points are None. Cannot calculate angle.")
                return None

            BA = pointA - pointB  # Vector from B to A
            BC = pointC - pointB  # Vector from B to C

            norm_BA = np.linalg.norm(BA)
            norm_BC = np.linalg.norm(BC)
            if norm_BA == 0 or norm_BC == 0:
                self.logger.warning("Zero length vector encountered. Cannot calculate angle.")
                return None

            cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle_rad = np.arccos(cosine_angle)
            angle_deg = np.degrees(angle_rad)
            self.logger.debug(f"Calculated joint angle: {angle_deg} degrees.")
            return angle_deg
        except Exception as e:
            self.logger.exception(f"Failed to calculate angle: {e}")
            return None

    def _save_frame_and_labels(self, frame, pose_landmarks, pose_world_landmarks, label_name, video_name, frames_output_dir, labels_data, joint_angles=None):
        """
        Saves the frame image and appends label data including the landmarks and joint angles.

        Args:
            frame (ndarray): The video frame.
            pose_landmarks (LandmarkList): Detected pose landmarks.
            pose_world_landmarks (LandmarkList): Detected world pose landmarks.
            label_name (str): Label associated with the frame.
            video_name (str): Name of the video file.
            frames_output_dir (str): Directory to save frames.
            labels_data (list): List to append label data dictionaries.
            joint_angles (dict, optional): Dictionary of joint angles.
        """
        try:
            frame_filename = f'{video_name}_frame_{self.global_frame_count:06d}.jpg'
            frame_path = os.path.join(frames_output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            self.logger.debug(f"Saved frame image: {frame_path}")

            label_entry = {'frame_filename': frame_filename, 'movement_label': label_name}

            for idx, lm in enumerate(pose_world_landmarks.landmark):
                label_entry[f'x{idx}'] = lm.x
                label_entry[f'y{idx}'] = lm.y
                label_entry[f'z{idx}'] = lm.z

            if joint_angles:
                for joint_name, angle in joint_angles.items():
                    label_entry[joint_name] = angle

            labels_data.append(label_entry)
            self.logger.debug(f"Appended label data for frame {self.global_frame_count}.")
        except Exception as e:
            self.logger.exception(f"Failed to save frame and labels for frame {self.global_frame_count}: {e}")


def main():
    """
    Main function to load configuration and start processing.
    """
    try:
        CONFIG_FILEPATH = 'labeller_config.yaml'
        if not os.path.exists(CONFIG_FILEPATH):
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILEPATH}")

        with open(CONFIG_FILEPATH, 'r') as file:
            config = yaml.safe_load(file)

        processor = BlazePoseVideoLabeller(config)
        processor.process_all_videos()

    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Failed to start BlazePoseVideoProcessor: {e}")


if __name__ == '__main__':
    main()
