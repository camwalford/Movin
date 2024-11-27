import time
from mapper import InputMapper
from camera import Camera
from labeller import Labeller
from classifier import Classifier
from device import ConnectedDevice


def run_app():
    # Get game from user input
    print("Game choices:", list(InputMapper.game_configs.keys()))
    while True:
        game = input("Please provide a game: ")
        if game in InputMapper.game_configs:
            break
        else:
            print("Invalid game.")

    # Setup classes
    mapper = InputMapper(game)
    camera = Camera()
    labeller = Labeller()
    classifier = Classifier("classifier.keras")
    device = ConnectedDevice()

    # Run pipeline
    while True:
        print("\nCapturing image...")
        image = camera.capture()
        landmarks = labeller.extract_landmarks(image)
        exercise, probability = classifier.predict(landmarks)
        print("Exercise identified:", exercise, ", Probability:", probability)
        key = mapper.exercise_to_key(exercise)
        device.execute(key)
        time.sleep(3)


if __name__ == "__main__":
    run_app()
