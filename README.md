
# Steps to Run the App

## Prerequisites
- The app is designed for **Python 3.8** and is optimized for the **NVIDIA Jetson Nano** or a **Linux environment**.
- Ensure that the **default camera** is configured and ready for use.
- **Important:** You will need **two terminal windows** running simultaneously:
  - One for starting the Bluetooth server and ensuring a Bluetooth device is connected.
  - One for running the application.

---

## Steps to Set Up the Environment

1. Create a virtual environment with Python 3.8:
   ```bash
   python3.8 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Alternatively**, manually install the required imports:
   ```bash
   pip install numpy~=1.24.3 opencv-python~=4.10.0.84 mediapipe~=0.10.9 pynput~=1.7.7 tensorflow~=2.13.0 dbus-python~=1.3.2 keyboard==0.13.5 evdev==1.7.1 PyGObject==3.50.0 pycairo==1.27.0
   ```

---

## Steps to Run the Bluetooth Server

1. **In the first terminal:**
   - Ensure the **default camera** is configured and operational.
   - Copy the D-Bus configuration file to the system directory:
     ```bash
     sudo cp ./src/jetson/bt_utils/org.yaptb.btkbservice.conf /etc/dbus-1/system.d
     ```

   - Disable the Bluetooth service:
     ```bash
     sudo systemctl disable bluetooth
     ```

   - Stop the Bluetooth service:
     ```bash
     sudo systemctl stop bluetooth
     ```

   - Start the Bluetooth server by running the script:
     ```bash
     sudo python start_bt_server.py
     ```

2. Verify that the **Bluetooth device** is connected and ready.
   - The app uses `dbus-python`, so your Bluetooth device should be properly paired and connected to the device where the key mappings will be sent.

---

## Steps to Run the Application

1. **In the second terminal:**
   - Activate the virtual environment:
     ```bash
     source venv/bin/activate
     ```

   - Navigate to the source directory for Jetson:
     ```bash
     cd src/jetson
     ```

   - Run the application:
     ```bash
     python app.py
     ```

---

## Notes
- The Bluetooth functionality was developed with the **NVIDIA Jetson Nano** in mind. Compatibility with other environments is not guaranteed.
- Ensure **both terminal windows remain open and running** while the application is in use.
