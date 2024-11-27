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
from src.utils.logger import setup_logger
import logging


class BlazePoseVideoProcessor:
    def __init__(self, config):
        """
        Initializes the BlazePoseVideoProcessor.

        Args:
            config (dict): Configuration dictionary.
        """
        self.logger = setup_logger(name="BlazePoseVideoProcessor", log_dir=config.get('log_dir', 'logs'),
                                   log_level=config.get('log_level', 'INFO'))
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
        self.global_frame_count = 0  # Counter for frames processed
        self.save_overlay_video = config.get('save_overlay_video', False)
        self.logger.info("BlazePoseVideoProcessor initialized successfully.")

    def process_all_videos(self):
        """
        Processes all videos in the input directory and saves outputs to Parquet files.
        """
        try:
            # Extract only the target label from the movement labels dictionary if multi_label is disabled
            if not self.multi_label and self.target_label:
                self.labels = {self.target_label: self.labels.get(self.target_label, {})}
                self.logger.info(f"Multi-label disabled. Target label set to: {self.target_label}")
            elif not self.multi_label and not self.target_label:
                self.logger.warning("Multi-label is disabled but no target_label is specified.")

            for label_name, label_data in self.labels.items():
                joint_angles = label_data.get('joints', [])
                self.logger.info(f"Processing label: {label_name} with {len(joint_angles)} joint angles.")

                # Process videos in both 'train' and 'test' directories
                for dataset_type in ['train', 'test']:
                    input_dataset_dir = os.path.join(self.input_dir, dataset_type, label_name)
                    output_dataset_dir = os.path.join(self.output_dir, dataset_type, label_name)

                    # Create timestamped subdirectory in the output dataset directory
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_dataset_dir = os.path.join(output_dataset_dir, timestamp)
                    frames_output_dir = os.path.join(output_dataset_dir, 'frames')
                    labels_parquet_path = os.path.join(output_dataset_dir, 'labels.parquet')
                    overlay_video_path = os.path.join(output_dataset_dir, f'overlay_{label_name}')

                    # Ensure output directories exist
                    os.makedirs(frames_output_dir, exist_ok=True)
                    self.logger.debug(f"Created output directories at {output_dataset_dir}")

                    # Initialize labels_data list
                    labels_data = []

                    # Initialize video writer if saving overlay video
                    overlay_writer = None
                    if self.save_overlay_video:
                        # We'll initialize the writer inside process_video to handle multiple videos
                        self.logger.info("Overlay video saving is enabled.")

                    # Process videos in the input dataset directory
                    video_paths = glob(os.path.join(input_dataset_dir, '*.mp4'))
                    if not video_paths:
                        self.logger.warning(f"No .mp4 files found in {input_dataset_dir}")
                        continue  # Move to the next dataset_type

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

                    # After processing all videos for this dataset_type and label_name, save labels_data
                    if labels_data:
                        df_labels = pd.DataFrame(labels_data)
                        try:
                            df_labels.to_parquet(labels_parquet_path, index=False)
                            self.logger.info(f"Saved labels to {labels_parquet_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to save Parquet file {labels_parquet_path}: {e}")
                    else:
                        self.logger.warning(f"No labels data collected for {label_name} in {dataset_type}.")

                    # Note: Overlay video handling is managed within process_video
                    self.logger.info(f"Finished processing label '{label_name}'.")

            self.pose.close()
            self.logger.info("Processed all videos successfully.")
        except Exception as e:
            self.logger.exception(f"An error occurred while processing videos: {e}")

    def process_video(self, video_path, frames_output_dir, labels_data, label_name, joint_angles, overlay_video_path):
        """
        Processes a single video to extract frames, landmarks, and optionally joint angles.
        Saves the output as frames and updates labels_data.
        Optionally saves an overlay video with landmarks and joint angles drawn.

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

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS not found
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.debug(
            f"Video '{video_name}' opened: {frame_width}x{frame_height} at {fps} FPS with {total_frames} frames.")

        # Initialize overlay video writer if enabled
        overlay_writer = None
        if self.save_overlay_video:
            try:
                overlay_video_path = f"{overlay_video_path}_{video_name}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (frame_width, frame_height))
                self.logger.debug(f"Overlay video writer initialized at {overlay_video_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize overlay video writer: {e}")

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

                    # Write the overlay frame to video if enabled
                    if self.save_overlay_video and overlay_writer is not None:
                        overlay_writer.write(processed_frame)
                        self.logger.debug(f"Wrote overlay frame {frame_index} to {overlay_video_path}")

                    pbar.update(1)
        except Exception as e:
            self.logger.exception(f"An error occurred while processing video '{video_path}': {e}")
        finally:
            cap.release()
            self.logger.debug(f"Released video capture for {video_path}")
            if overlay_writer is not None:
                overlay_writer.release()
                self.logger.debug(f"Released overlay video writer for {overlay_video_path}")

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
            frame_height, frame_width = frame.shape[:2]
            # Convert the frame to RGB as MediaPipe BlazePose expects RGB input
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            self.logger.debug("Pose processing completed for current frame.")

            # Make a copy of the frame to draw overlays
            overlay_frame = frame.copy()
            frame_saved = False
            angles_display = {}  # To store angles for display

            if results.pose_landmarks:
                self._draw_landmarks(overlay_frame, results.pose_landmarks)
                self.logger.debug("Pose landmarks drawn on frame.")

                # Calculate joint angles
                angles_dict = self._calculate_joint_angles(results.pose_landmarks, joint_angles, frame_width, frame_height)
                self._overlay_joint_angles(overlay_frame, results.pose_landmarks, angles_dict, frame_width, frame_height)
                self.logger.debug("Joint angles calculated and overlaid.")

                # Check if angles match thresholds
                angles_match = True
                if self.use_joint_angles:
                    angles_match = self._check_joint_angle_thresholds(angles_dict, joint_angles)
                self.logger.debug(f"Joint angles match: {angles_match}")

                # Display match status
                self._display_label(overlay_frame, label_name, match=angles_match)

                # Save frame and labels if angles match
                if angles_match:
                    self._save_frame_and_labels(
                        frame=frame,
                        pose_landmarks=results.pose_landmarks,
                        label_name=label_name,
                        video_name=video_name,
                        frames_output_dir=frames_output_dir,
                        labels_data=labels_data,
                        joint_angles=angles_dict
                    )
                    frame_saved = True
                    self.logger.info(f"Frame {self.global_frame_count} matched and saved.")
                else:
                    self.logger.debug(f"Frame {self.global_frame_count} did not match label '{label_name}'.")
            else:
                # If no pose is detected, log and optionally handle accordingly
                self.logger.debug(f"No pose detected in frame {self.global_frame_count} of video '{video_name}'.")

            return overlay_frame
        except Exception as e:
            self.logger.exception(f"An error occurred while processing frame {self.global_frame_count}: {e}")
            return frame  # Return the original frame if processing fails

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
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2)
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
                # Extract side and joint
                parts = joint_name.split('_')
                joint = parts[0]
                side = parts[1] if len(parts) > 1 else 'BOTH'

                # Determine the landmark to position the angle text
                if joint.upper() == "KNEE":
                    landmark_name = 'RIGHT_KNEE' if 'RIGHT' in side else 'LEFT_KNEE'
                elif joint.upper() == "ELBOW":
                    landmark_name = 'RIGHT_ELBOW' if 'RIGHT' in side else 'LEFT_ELBOW'
                elif joint.upper() == "SHOULDER":
                    landmark_name = 'RIGHT_SHOULDER' if 'RIGHT' in side else 'LEFT_SHOULDER'
                elif joint.upper() == "HIP":
                    landmark_name = 'RIGHT_HIP' if 'RIGHT' in side else 'LEFT_HIP'
                else:
                    # Default to a central landmark if joint is unknown
                    landmark_name = 'NOSE'

                # Get the landmark coordinates
                try:
                    coords = self._get_coords(pose_landmarks, landmark_name, frame_width, frame_height)
                    x, y = int(coords[0]), int(coords[1])

                    # Define the text to overlay
                    angle_text = f"{joint}: {int(angle)}°"

                    # Define position offset
                    offset_x = 20
                    offset_y = -20

                    # Put the text on the frame
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
                except KeyError:
                    self.logger.warning(f"Landmark '{landmark_name}' not found for overlaying angle '{joint_name}'.")
                except Exception as e:
                    self.logger.error(f"Failed to overlay angle '{joint_name}': {e}")
        except Exception as e:
            self.logger.error(f"Failed to overlay joint angles: {e}")

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
                angle_left = None
                angle_right = None

                # Calculate angles based on joint type
                if name.upper() == "KNEE":
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
                elif name.upper() == "ELBOW":
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
                elif name.upper() == "SHOULDER":
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
                elif name.upper() == "HIP":
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
                    continue  # Skip this joint

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
                    f"Checking angle thresholds for joint '{name}' on side '{side}' with thresholds [{min_angle}, {max_angle}].")

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

            # All joints passed the threshold checks
            self.logger.debug("All joint angles matched the thresholds.")
            return True
        except Exception as e:
            self.logger.exception(f"An error occurred while checking joint angle thresholds: {e}")
            return False

    def _get_coords(self, pose_landmarks, landmark_name, frame_width, frame_height):
        try:
            landmark = pose_landmarks.landmark[PoseLandmark[landmark_name.upper()].value]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            # z = landmark.z * frame_width  # Uncomment if you want to scale z-coordinate
            z = landmark.z  # Using the normalized z-coordinate as provided
            coords = np.array([x, y])
            # coords = np.array([x, y, z])  # Use this if including z in calculations
            self.logger.debug(f"Retrieved coordinates for {landmark_name}: {coords}")
            return coords
        except KeyError:
            self.logger.error(f"Landmark '{landmark_name}' not found in PoseLandmark.")
            return None
        except Exception as e:
            self.logger.exception(f"An error occurred while getting coordinates for '{landmark_name}': {e}")
            return None

    def _calculate_angle(self, pointA, pointB, pointC):
        """ Calculates the angle at pointB formed by the line segments BA and BC.
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
            # Correct vector definitions
            BA = pointA - pointB  # Vector from B to A
            BC = pointC - pointB  # Vector from B to C

            # Compute the angle between BA and BC
            norm_BA = np.linalg.norm(BA)
            norm_BC = np.linalg.norm(BC)
            if norm_BA == 0 or norm_BC == 0:
                self.logger.warning("Zero length vector encountered. Cannot calculate angle.")
                return None
            cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure within valid range
            angle_rad = np.arccos(cosine_angle)
            angle_deg = np.degrees(angle_rad)
            joint_angle = angle_deg  # Use angle directly, as it's now correct
            self.logger.debug(f"Calculated joint angle: {joint_angle} degrees.")
            return joint_angle
        except Exception as e:
            self.logger.exception(f"Failed to calculate angle: {e}")
            return None

    def _save_frame_and_labels(self, frame, pose_landmarks, label_name, video_name, frames_output_dir, labels_data, joint_angles=None):
        """
        Saves the frame image and appends label data including the landmarks.

        Args:
            frame (ndarray): The video frame.
            pose_landmarks (LandmarkList): Detected pose landmarks.
            label_name (str): Label associated with the frame.
            video_name (str): Name of the video file.
            frames_output_dir (str): Directory to save frames.
            labels_data (list): List to append label data dictionaries.
            joint_angles (dict, optional): Dictionary of joint angles.
        """
        try:
            # Save the original frame image (without landmarks)
            frame_filename = f'{video_name}_frame_{self.global_frame_count:06d}.jpg'
            frame_path = os.path.join(frames_output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            self.logger.debug(f"Saved frame image: {frame_path}")

            # Prepare the label data dictionary with landmarks
            label_entry = {'frame_filename': frame_filename, 'movement_label': label_name}

            # Add the landmarks to the label_entry
            for idx, lm in enumerate(pose_landmarks.landmark):
                label_entry[f'x{idx + 1}'] = lm.x
                label_entry[f'y{idx + 1}'] = lm.y
                label_entry[f'z{idx + 1}'] = lm.z  # Include z-coordinate in labels data

            # Add joint angles to the label_entry if available
            if joint_angles:
                for joint_name, angle in joint_angles.items():
                    label_entry[joint_name] = angle

            # Append the label data to the labels_data list
            labels_data.append(label_entry)
            self.logger.debug(f"Appended label data for frame {self.global_frame_count}.")
        except Exception as e:
            self.logger.exception(f"Failed to save frame and labels for frame {self.global_frame_count}: {e}")

def main():
    """
    Main function to load configuration and start processing.
    """
    try:
        # Centralize configuration in a YAML file
        CONFIG_FILEPATH = 'labeller_config.yaml'
        if not os.path.exists(CONFIG_FILEPATH):
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILEPATH}")

        with open(CONFIG_FILEPATH, 'r') as file:
            config = yaml.safe_load(file)

        processor = BlazePoseVideoProcessor(config)
        processor.process_all_videos()

    except Exception as e:
        # Since logging is initialized inside the class, we need to set up a temporary logger here
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Failed to start BlazePoseVideoProcessor: {e}")

if __name__ == '__main__':
    main()