from modules.camera import Camera
from modules.labeller import Labeller
from modules.classifier import Classifier
from modules.mapper import InputMapper
from modules.detector import MovementDetector

# Classifiers
classifiers = {
    "cam": ["models/model.h5", "models/label_encoder_h5.npy"],
    "kate": ["models/classifier.keras"],
}

# App config
config = {
    "game": "2048",  # Leave blank to get user's input
    "classifier": classifiers["cam"],
    "detector": {"queue_size": 30, "threshold": 2, "z_weight": 1},
    "showDisplay": True,
    "keyboard": "default"
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
    camera = Camera()
    labeller = Labeller()
    detector = MovementDetector(**config["detector"])
    classifier = Classifier(*config["classifier"])
    mapper = InputMapper(game)
    keyboard = None


    # from modules.bt_utils.bluetoothkeyboard import BluetoothKeyboard

    # keyboard = BluetoothKeyboard()

    exercise = "idle"
    next_input = True  # Flag to check if the next input is valid

    # Run pipeline
    while True:
        # CAPTURE IMAGE
        image = camera.capture()

        # EXTRACT LANDMARKS
        flattened_landmarks, non_flattened_landmarks = labeller.extract_landmarks(image)
        if flattened_landmarks is None or not flattened_landmarks.any():
            # DISPLAY CAMERA
            if config["showDisplay"]:
                camera.display(image, "No player detected.")
            continue

        # CLASSIFY EXERCISE
        if detector.movement_detected(non_flattened_landmarks):
            exercise, probability = classifier.predict(flattened_landmarks.reshape(1, -1))
            # exercise, probability = "idle", 0.1
            print("Exercise identified:", exercise, "\nProbability:", probability)
            if exercise != "idle" and next_input:
                key = mapper.exercise_to_key(exercise)
                # keyboard.send_string(key)
                next_input = False
            if exercise == "idle":
                next_input = True

        # DISPLAY CAMERA
        if config["showDisplay"]:
            camera.display(image, exercise)


if __name__ == "__main__":
    run_app()

