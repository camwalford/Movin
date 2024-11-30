import time
from jetsoncamera import JetsonCamera, LaptopCamera
from labeller import Labeller
from classifier import Classifier
from mapper import InputMapper
from device import ConnectedDevice
from src.jetson.detector import MovementDetector


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
    camera = LaptopCamera()
    labeller = Labeller()
    detector = MovementDetector(queue_size=30, threshold=10)
    classifier = Classifier("classifier.keras")
    mapper = InputMapper(game)
    device = ConnectedDevice()

    # Run pipeline
    while True:
        # print("\nCapturing image...")
        image = camera.capture()
        flattened_landmarks, non_flattened_landmarks = labeller.extract_landmarks(image)
        if flattened_landmarks is None or not flattened_landmarks.any():
            # print("No landmarks detected. Skipping...")
            continue
        if detector.movement_detected(non_flattened_landmarks):
            exercise, probability = classifier.predict(flattened_landmarks.reshape(1, -1))
            print("Exercise identified:", exercise, ", Probability:", probability)
            key = mapper.exercise_to_key(exercise)
            device.execute(key)
            time.sleep(1)


if __name__ == "__main__":
    run_app()
