class InputMapper:

    game_configs = {
        "2048": {
            "left_lunge": "%left%",
            "right_lunge": "%right%",
            "jumping_jacks": "%up%",
            "squat": "%down%",
        }
    }

    def __init__(self, game):
        if game not in self.game_configs:
            raise Exception("Invalid game.")
        self.config = InputMapper.game_configs[game]

    def exercise_to_key(self, exercise):
        return self.config[exercise]

