import os
import sys
import threading
import subprocess
from .hid_server import BTKbService
from gi.repository import GLib
from dbus.mainloop.glib import DBusGMainLoop
import time


class BluetoothDaemon:
    """Handles the Bluetooth daemon process."""

    def __init__(self):
        self.process = None

    def start(self):
        try:
            # Start the Bluetooth daemon process
            self.process = subprocess.Popen(
                ["sudo", "/usr/lib/bluetooth/bluetoothd", "-P", "input"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Bluetooth daemon started with PID: {0}".format(self.process.pid))
            # Give the daemon some time to initialize
            time.sleep(2)
        except FileNotFoundError:
            print("The bluetoothd executable could not be found. Check the path.")
            self.process = None

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Bluetooth daemon terminated.")


class BluetoothServer:
    """Handles the Bluetooth server and GLib main loop."""

    def __init__(self):
        self.mainloop = None
        self.myservice = None

    def start(self):
        if not os.geteuid() == 0:
            sys.exit('Only root can run this script')

        # Initialize DBus and GLib
        DBusGMainLoop(set_as_default=True)
        self.myservice = BTKbService()  # Assuming BTKbService is defined elsewhere
        self.mainloop = GLib.MainLoop()

        # Run the main loop in a separate thread
        try:
            thread = threading.Thread(target=self.run_mainloop)
            thread.daemon = True  # Daemonize the thread
            thread.start()
            print("Bluetooth server initialized and running in the background.")
        except Exception as e:
            print("Failed to start Bluetooth server: {0}".format(e))

    def run_mainloop(self):
        try:
            self.mainloop.run()
        except Exception as e:
            print("GLib MainLoop error: {0}".format(e))

    def stop(self):
        if self.mainloop:
            self.mainloop.quit()
            print("Bluetooth server stopped.")


class BTHandler:
    """Handles both the Bluetooth daemon and server."""

    def __init__(self):
        self.daemon = BluetoothDaemon()
        self.server = BluetoothServer()

    def start(self):
        try:
            print("Starting Bluetooth Daemon...")
            self.daemon.start()

            print("Starting Bluetooth Server...")
            self.server.start()
        except Exception as e:
            print("Error starting BTHandler: {0}".format(e))
            self.stop()

    def stop(self):
        print("Stopping BTHandler...")
        try:
            self.server.stop()
        except Exception as e:
            print("Error stopping Bluetooth server: {0}".format(e))

        try:
            self.daemon.stop()
        except Exception as e:
            print("Error stopping Bluetooth daemon: {0}".format(e))


if __name__ == "__main__":
    handler = BTHandler()

    try:
        print("Starting Bluetooth Handler...")
        handler.start()
        print("Bluetooth Handler is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # Keep the program alive
    except KeyboardInterrupt:
        print("\nStopping Bluetooth Handler...")
    finally:
        handler.stop()
