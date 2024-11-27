class InputMapper:

    def __init__(self, game):
        self.game = game

    def exercise_to_key(self, exercise):
        print("InputMapper received exercise:", exercise)
        print("Mapping to key.")
        return "this is a key"

