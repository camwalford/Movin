from pynput.keyboard import Key

# A simplified keymap for pynput.
# Letters and numbers can just be the literal string,
# special keys map to Key.* constants.

pynput_keymap = {
    "KEY_A": 'a',
    "KEY_B": 'b',
    "KEY_C": 'c',
    "KEY_D": 'd',
    "KEY_E": 'e',
    "KEY_F": 'f',
    "KEY_G": 'g',
    "KEY_H": 'h',
    "KEY_I": 'i',
    "KEY_J": 'j',
    "KEY_K": 'k',
    "KEY_L": 'l',
    "KEY_M": 'm',
    "KEY_N": 'n',
    "KEY_O": 'o',
    "KEY_P": 'p',
    "KEY_Q": 'q',
    "KEY_R": 'r',
    "KEY_S": 's',
    "KEY_T": 't',
    "KEY_U": 'u',
    "KEY_V": 'v',
    "KEY_W": 'w',
    "KEY_X": 'x',
    "KEY_Y": 'y',
    "KEY_Z": 'z',
    "KEY_1": '1',
    "KEY_2": '2',
    "KEY_3": '3',
    "KEY_4": '4',
    "KEY_5": '5',
    "KEY_6": '6',
    "KEY_7": '7',
    "KEY_8": '8',
    "KEY_9": '9',
    "KEY_0": '0',
    "KEY_ENTER": Key.enter,
    "KEY_SPACE": Key.space,
    "KEY_TAB": Key.tab,
    "KEY_LEFT": Key.left,
    "KEY_RIGHT": Key.right,
    "KEY_UP": Key.up,
    "KEY_DOWN": Key.down,
    "KEY_ESC": Key.esc,
    "KEY_BACKSPACE": Key.backspace,
    "KEY_DELETE": Key.delete,
    "KEY_INSERT": Key.insert,
    "KEY_HOME": Key.home,
    "KEY_END": Key.end,
    "KEY_PAGEUP": Key.page_up,
    "KEY_PAGEDOWN": Key.page_down,
    "KEY_F1": Key.f1,
    "KEY_F2": Key.f2,
    "KEY_F3": Key.f3,
    "KEY_F4": Key.f4,
    "KEY_F5": Key.f5,
    "KEY_F6": Key.f6,
    "KEY_F7": Key.f7,
    "KEY_F8": Key.f8,
    "KEY_F9": Key.f9,
    "KEY_F10": Key.f10,
    "KEY_F11": Key.f11,
    "KEY_F12": Key.f12,
    # You can map more as needed...
}

# Modifier keys in pynput:
# Key.shift, Key.ctrl, Key.alt, Key.cmd (or Key.cmd_l, Key.cmd_r, etc.)
# If you want to handle modifiers, you can map them similarly:
pynput_modmap = {
    "KEY_LEFTSHIFT": Key.shift,
    "KEY_LEFTCTRL": Key.ctrl,
    "KEY_LEFTALT": Key.alt,
    "KEY_LEFTMETA": Key.cmd,
    "KEY_RIGHTSHIFT": Key.shift,
    "KEY_RIGHTCTRL": Key.ctrl,
    "KEY_RIGHTALT": Key.alt,
    "KEY_RIGHTMETA": Key.cmd,
}


def pynput_convert(key_name):
    """
    Convert a KEY_* name to a pynput-compatible key event.
    Returns either a string or a Key constant.
    Returns None if no mapping is found.
    """
    return pynput_keymap.get(key_name, None)


def pynput_modkey(key_name):
    """
    Convert a KEY_* modifier to a pynput Key modifier.
    Returns the modifier Key if found, else None.
    """
    return pynput_modmap.get(key_name, None)
