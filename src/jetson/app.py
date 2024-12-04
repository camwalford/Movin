import time
from jetsoncamera import JetsonCamera, LaptopCamera
from labeller import Labeller
from classifier import Classifier
from mapper import InputMapper
from device import ConnectedDevice
from detector import MovementDetector

cam_classifier = Classifier("models/model.h5", "models/label_encoder_h5.npy")
# kate_classifier = Classifier("src/jetson/models/classifier.keras")
def run_app():
    # Get game from user input
    # print("Game choices:", list(InputMapper.game_configs.keys()))
    # while True:
    #     game = input("Please provide a game: ")
    #     if game in InputMapper.game_configs:
    #         break
    #     else:
    #         print("Invalid game.")

    # Setup classes
    camera = LaptopCamera()
    labeller = Labeller()
    detector = MovementDetector(queue_size=30, threshold=4, z_weight=1)
    classifier = cam_classifier
    mapper = InputMapper(game)
    device = ConnectedDevice()

    exercise = "idle"
    next_input = True # Flag to check if the next input is valid
    display = True # Set to True to display the camera feed
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
            # exercise, probability = "idle", 0.1
            print("Exercise identified:", exercise,
                  "\nProbability:", probability)
            if exercise != "idle" and next_input:
                key = mapper.exercise_to_key(exercise)
                device.execute(key)
                next_input = False
            if exercise == "idle":
                next_input = True

        if display:
            camera.display(image, exercise)

if __name__ == "__main__":
    run_app()
