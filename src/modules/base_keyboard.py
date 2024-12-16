

# src/modules/base_keyboard.py
import time

class BaseKeyboard:
    def __init__(self, target_length=6):
        self.target_length = target_length
        self.mod_keys = 0b00000000
        self.pressed_keys = []
        # keymap might be shared or provided by subclasses
        # self.keymap = ...

    def update_mod_keys(self, mod_key, value):
        """
        Update the modifier keys (e.g., Shift, Ctrl).
        """
        bit_mask = 1 << (7 - mod_key)
        if value:
            self.mod_keys |= bit_mask
        else:
            self.mod_keys &= ~bit_mask

    def update_keys(self, norm_key, value):
        """
        Update the list of currently pressed keys.
        """
        if value < 1:  # Key released
            if norm_key in self.pressed_keys:
                self.pressed_keys.remove(norm_key)
        elif norm_key not in self.pressed_keys:  # Key pressed
            self.pressed_keys.insert(0, norm_key)
        len_delta = self.target_length - len(self.pressed_keys)
        if len_delta < 0:
            self.pressed_keys = self.pressed_keys[:len_delta]
        elif len_delta > 0:
            self.pressed_keys.extend([0] * len_delta)

    def parse_text(self, text):
        """
        Parse the input text to identify special keys (e.g., %up% -> KEY_UP).
        """
        tokens = []
        temp = ""
        special = False

        for char in text:
            if char == "%":
                if special:  # End of a special token
                    tokens.append("KEY_{0}".format(temp.upper()))
                    temp = ""
                special = not special
            elif special:
                temp += char
            else:
                tokens.append(char)

        return tokens

    def send_string(self, text):
        """
        Send a predefined string as a series of keypress events.
        This method relies on self.send_key_event(norm_key, value) which must
        be implemented by the subclass.
        """
        tokens = self.parse_text(text)

        for token in tokens:
            if token.startswith("KEY_"):  # Handle special keys
                norm_key = self.convert_key(token)
                if norm_key > -1:
                    self.update_keys(norm_key, 1)  # Key press
                    self.send_current_keys()
                    time.sleep(0.05)
                    self.update_keys(norm_key, 0)  # Key release
                    self.send_current_keys()
                    time.sleep(0.05)
            else:  # Regular characters
                key_name = "KEY_{0}".format(token.upper()) if token.isalpha() else "KEY_{0}".format(token)
                if self.is_valid_key(key_name):
                    norm_key = self.convert_key(key_name)
                    self.update_keys(norm_key, 1)
                    self.send_current_keys()
                    time.sleep(0.05)
                    self.update_keys(norm_key, 0)
                    self.send_current_keys()
                    time.sleep(0.05)

    def is_valid_key(self, key_name):
        # Check if the key_name is in the keymap. Subclass or keymap can define this.
        return key_name in self.keymap

    def convert_key(self, key_name):
        # Convert a key name (e.g. "KEY_A") to its code using self.keymap
        # To be implemented by subclass or assume keymap is global.
        return self.keymap.convert(key_name)

    def send_current_keys(self):
        """
        Send the current key states. This is an abstract method.
        Each subclass must implement how keys are actually "sent" to the system.
        """
        raise NotImplementedError("Subclasses must implement send_current_keys().")


