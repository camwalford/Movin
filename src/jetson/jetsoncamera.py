import cv2


class JetsonCamera:

    def __init__(self, width=640, height=480, fps=60):
        # Open the default camera
        self.camera = cv2.VideoCapture(0)

        # Set resolution
        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)

        else:
            self.camera = None
            print("Could not open camera.")

    def capture(self, output_path=""):
        # Handle uninitialized camera
        if self.camera is None:
            print("Camera not initialized.")

        # Capture a single frame
        ret, frame = self.camera.read()

        if output_path:
            cv2.imwrite(output_path, frame)

        if ret:
            return frame  # Return as NumPy array
        else:
            print("Could not capture image.")

    def display(self, image, exercise="idle", window_name="Camera"):
        color = (255, 0, 0) if exercise == "idle" else (0, 255, 0)
        cv2.putText(image, f"Exercise: {exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            cv2.destroyAllWindows()  # Optionally close any OpenCV windows.

    def __del__(self):
        self.release()


if __name__ == "__main__":
    camera = JetsonCamera()
    camera.capture("image.jpg")

