# src/modules/bluetooth_keyboard.py
import dbus

from src.modules.base_keyboard import BaseKeyboard
from src.modules.bt_utils import keymap

class BluetoothKeyboard(BaseKeyboard):
    HID_DBUS = 'org.yaptb.btkbservice'
    HID_SRVC = '/org/yaptb/btkbservice'

    def __init__(self):
        super().__init__(target_length=6)
        self.bus = dbus.SystemBus()
        self.btkobject = self.bus.get_object(BluetoothKeyboard.HID_DBUS, BluetoothKeyboard.HID_SRVC)
        self.btk_service = dbus.Interface(self.btkobject, BluetoothKeyboard.HID_DBUS)
        self.keymap = keymap  # Assume keymap is a module with a convert method.

    @property
    def state(self):
        return [0xA1, 0x01, self.mod_keys, 0, *self.pressed_keys]

    def send_current_keys(self):
        self.btk_service.send_keys(self.state)
