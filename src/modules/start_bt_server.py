import time
from bt_utils.btserver import BTHandler


def main():
    bts = BTHandler()
    bts.start()
    while True:
        time.sleep(3600)  # Infinite loop


if __name__ == "__main__":
    main()
