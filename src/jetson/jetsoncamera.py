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


    def display(self, image, exercise="idle", window_name="Camera"):
        color = (255, 0, 0) if exercise == "idle" else (0, 255, 0)
        cv2.putText(image, f"Exercise: {exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)