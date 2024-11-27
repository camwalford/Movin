import time
from mapper import InputMapper
from camera import Camera
from labeller import Labeller
from classifier import Classifier
from device import ConnectedDevice


def run_app():
    game = input("Please provide a game: ")

    mapper = InputMapper(game)
    camera = Camera()
    labeller = Labeller()
    classifier = Classifier("classifier.keras")
    device = ConnectedDevice()

    while True:
        print("\nCapturing image...")
        image = camera.capture()
        landmarks = labeller.extract_landmarks(image)
        exercise = classifier.predict(landmarks)
        print("Exercise identified:", exercise)
        key = mapper.exercise_to_key(exercise)
        device.execute(key)
        time.sleep(3)


if __name__ == "__main__":
    run_app()
