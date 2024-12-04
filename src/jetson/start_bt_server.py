from bt_utils.btserver import BTHandler
from time import sleep

def main():
    bts = BTHandler()
    bts.start()

if __name__ == "__main__":
    main()
