from jetsoncamera import JetsonCamera, LaptopCamera
from labeller import Labeller
from classifier import Classifier
from mapper import InputMapper
from device import ConnectedDevice
from detector import MovementDetector

# Classifiers
classifiers = {
    "cam": ["models/model.h5", "models/label_encoder_h5.npy"],
    "kate": ["models/classifier.keras"],
}

# App config
config = {
    "game": "2048",  # Leave blank to get user's input
    "classifier": classifiers["cam"],
    "showDisplay": True
}


def run_app():
    # Get game from user input
    game = config["game"]
    if game is None:
        print("Game choices:", list(InputMapper.game_configs.keys()))
        while True:
            game = input("Please provide a game: ")
            if game in InputMapper.game_configs:
                break
            else:
                print("Invalid game.")

    # Setup classes
    camera = JetsonCamera()
    labeller = Labeller()
    detector = MovementDetector(queue_size=30, threshold=4, z_weight=1)
    classifier = Classifier(*config["classifier"])
    mapper = InputMapper(game)
    device = ConnectedDevice()

    exercise = "idle"
    next_input = True  # Flag to check if the next input is valid

    # Run pipeline
    while True:
        # CAPTURE IMAGE
        image = camera.capture()

        # EXTRACT LANDMARKS
        flattened_landmarks, non_flattened_landmarks = labeller.extract_landmarks(image)
        if flattened_landmarks is None or not flattened_landmarks.any():
            # print("No landmarks detected. Skipping...")
            continue

        # CLASSIFY EXERCISE
        if detector.movement_detected(non_flattened_landmarks):
            exercise, probability = classifier.predict(flattened_landmarks.reshape(1, -1))
            # exercise, probability = "idle", 0.1
            print("Exercise identified:", exercise, "\nProbability:", probability)
            if exercise != "idle" and next_input:
                key = mapper.exercise_to_key(exercise)
                device.execute(key)
                next_input = False
            if exercise == "idle":
                next_input = True

        # DISPLAY CAMERA
        if config["showDisplay"]:
            camera.display(image, exercise)


if __name__ == "__main__":
    run_app()

