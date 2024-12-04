import cv2


import threading

class JetsonCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.frame = None
        self.running = True

        if self.camera.isOpened():
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        else:
            self.camera = None
            print("Could not open camera.")

    def update(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame

    def capture(self, output_path=""):
        if self.frame is not None and output_path:
            cv2.imwrite(output_path, self.frame)
        return self.frame

    def display(self, exercise="idle", window_name="Camera"):
        if self.frame is not None:
            color = (255, 0, 0) if exercise == "idle" else (0, 255, 0)
            image = self.frame.copy()
            cv2.putText(image, f"Exercise: {exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow(window_name, image)
            cv2.waitKey(1)

    def release(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            cv2.destroyAllWindows()

    def __del__(self):
        self.release()



if __name__ == "__main__":
    camera = JetsonCamera()
    camera.capture("image.jpg")

