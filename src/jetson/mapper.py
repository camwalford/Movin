from pynput.keyboard import Key, Controller

class InputMapper:

    game_configs = {
        "2048": {
            "left": Key.left,
            "right": Key.right,
            "up": Key.up,
            "down": Key.down,
        }
    }

    def __init__(self, game):
        if game not in self.game_configs:
            raise Exception("Invalid game.")
        self.config = InputMapper.game_configs[game]

    def exercise_to_key(self, exercise):
        print("InputMapper received exercise:", exercise)
        print("Mapping to key.")
        return "this is a key"

