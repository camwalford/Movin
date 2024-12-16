# pynput_keyboard.py
from pynput.keyboard import Controller, Key
from time import sleep
from .base_keyboard import BaseKeyboard
from .pynput_keymap import pynput_convert

class PynputKeyboard(BaseKeyboard):
    def __init__(self):
        super().__init__(target_length=6)
        self.keyboard = Controller()

    def send_string(self, text):
        """
        Override send_string to directly press and release keys using pynput.
        This doesn't rely on the state-based logic of BaseKeyboard. It simply
        interprets tokens and simulates the key presses.
        """
        tokens = self.parse_text(text)

        for token in tokens:
            # If token is a special key token (e.g., KEY_UP, KEY_ENTER),
            # parse as-is. Otherwise, convert character to a KEY_* form.
            if token.startswith("KEY_"):
                key_name = token
            else:
                if token.isalpha():
                    key_name = f"KEY_{token.upper()}"
                else:
                    # For non-alpha chars (space, punctuation), you may need a mapping.
                    # For example, space -> KEY_SPACE
                    # If token == ' ', then key_name = 'KEY_SPACE'
                    key_name = f"KEY_{token.strip().upper()}" if token.strip() else "KEY_SPACE"

            key = pynput_convert(key_name)
            if key is not None:
                print(f"Pressing {key_name} using pynput")
                self.keyboard.press(key)
                sleep(0.05)
                self.keyboard.release(key)
                sleep(0.05)
