import dbus
from . import keymap
from time import sleep



class Keyboard:
    """
    Send a predefined string as HID messages over the keyboard D-Bus server.
    """

    HID_DBUS = 'org.yaptb.btkbservice'
    HID_SRVC = '/org/yaptb/btkbservice'

    def __init__(self):
        self.target_length = 6
        self.mod_keys = 0b00000000
        self.pressed_keys = []
        self.bus = dbus.SystemBus()
        self.btkobject = self.bus.get_object(Keyboard.HID_DBUS, Keyboard.HID_SRVC)
        self.btk_service = dbus.Interface(self.btkobject, Keyboard.HID_DBUS)

    def update_mod_keys(self, mod_key, value):
        """
        Update the modifier keys (e.g., Shift, Ctrl).
        """
        bit_mask = 1 << (7 - mod_key)
        if value:  # Set bit
            self.mod_keys |= bit_mask
        else:  # Clear bit
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

    @property
    def state(self):
        """
        Generate the HID message for the current key state.
        """
        return [0xA1, 0x01, self.mod_keys, 0, *self.pressed_keys]

    def send_keys(self):
        """
        Send the current HID state over D-Bus.
        """
        self.btk_service.send_keys(self.state)

    def send_string(self, text):
        """
        Send a predefined string as a series of keypress events.
        """
        tokens = self.parse_text(text)

        for token in tokens:
            if token.startswith("KEY_"):  # Handle special keys
                norm_key = keymap.convert(token)
                if norm_key > -1:
                    print("pressing")
                    self.update_keys(norm_key, 1)  # Key press
                    self.send_keys()
                    print("pressed")
                    sleep(0.05)  # Small delay for key press
                    print("releasing")
                    self.update_keys(norm_key, 0)  # Key release
                    self.send_keys()
                    print("released")
                    sleep(0.05)  # Small delay before next key
            else:  # Handle regular characters
                key_name = f"KEY_{token.upper()}" if token.isalpha() else f"KEY_{token}"
                if key_name in keymap.keytable:
                    norm_key = keymap.convert(key_name)
                    print("pressing")
                    self.update_keys(norm_key, 1)  # Key press
                    self.send_keys()
                    print("pressed")
                    sleep(0.05)  # Small delay for key press
                    print("releasing")
                    self.update_keys(norm_key, 0)  # Key release
                    self.send_keys()
                    print("released")
                    sleep(0.05)

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
                    tokens.append(f"KEY_{temp.upper()}")
                    temp = ""
                special = not special
            elif special:
                temp += char
            else:
                tokens.append(char)

        return tokens


if __name__ == '__main__':
    kb = BTKeyboard()

    print('Sending predefined string: "hello %up% %down% %left% %right%"')
    # kb.send_string("%up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% %up% %down% %left% %right% ")
    kb.send_string("%up% %down% %left% %right%")
    # kb.send_string("This%space%was%space%sent%space%using%space%a%space%bluetooth%space%hackerman%space%device%space%get%space%pwnd%space%%enter%")
    print('String sent successfully.')
