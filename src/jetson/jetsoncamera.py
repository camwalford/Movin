import cv2


class JetsonCamera:

    def __init__(self):
        print("Setting up camera.")

    def capture(self):
        print("Capturing image.")
        return "this is an image"


class LaptopCamera:

    def __init__(self):
        # Open the default camera
        self.camera = cv2.VideoCapture(0)

        # Handle unopened camera
        if not self.camera.isOpened():
            self.camera = None
            raise Exception("Could not open camera.")

    def capture(self, output_path=""):
        # Handle uninitialized camera
        if self.camera is None:
            raise Exception("Camera not initialized.")

        # Capture a single frame
        ret, frame = self.camera.read()

        if output_path:
            cv2.imwrite(output_path, frame)

        if ret:
            return frame  # Return as NumPy array
        else:
            raise Exception("Could not capture image.")
