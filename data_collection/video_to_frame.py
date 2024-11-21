import cv2
import os


class VideoToFrames:
    def __init__(self, video_path, output_dir, frame_rate=1):
        """
        Initializes the video to frames processor.

        :param video_path: Path to the input video file.
        :param output_dir: Directory to save the extracted frames.
        :param frame_rate: Number of frames to extract per second (default: 1 frame per second).
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_rate = frame_rate

    def extract_frames(self):
        """Extracts frames from the video and saves them as images."""
        # Check if the video file exists
        if not os.path.exists(self.video_path):
            print(f"Error: Video file '{self.video_path}' does not exist.")
            return

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Open the video file
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file '{self.video_path}'.")
            return

        # Get the video's frames per second (fps)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.frame_rate)

        # Initialize frame count
        frame_count = 0
        saved_count = 0

        print(f"Processing video: {self.video_path}")
        print(f"Saving frames to: {self.output_dir}")
        print(f"Extracting 1 frame every {frame_interval} frames ({self.frame_rate} fps).")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Save every nth frame based on the frame interval
            if frame_count % frame_interval == 0:
                frame_file = os.path.join(self.output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Extraction complete. Total frames saved: {saved_count}")


if __name__ == "__main__":
    # Replace with the path to your video and desired output directory
    VIDEO_PATH = "input_video.mkv"
    OUTPUT_DIR = "frames"
    FRAME_RATE = 30  # Number of frames per second to extract

    processor = VideoToFrames(VIDEO_PATH, OUTPUT_DIR, FRAME_RATE)
    processor.extract_frames()
