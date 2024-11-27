from pynput.keyboard import Key, Controller

class InputMapper:

    game_configs = {
        "2048": {
            "left_lunge": Key.left,
            "right_lunge": Key.right,
            "jumping_jacks": Key.up,
            "squat": Key.down,
        }
    }

    def __init__(self, game):
        if game not in self.game_configs:
            raise Exception("Invalid game.")
        self.config = InputMapper.game_configs[game]

    def exercise_to_key(self, exercise):
        return self.config[exercise]

