import sys, os
sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
from PyQt5.QtWidgets import QApplication
from pyqt.screen_calibration_widget import ManualScreen
        
def main():
    app = QApplication([])
    window = ManualScreen()
    window.delayedStart()
    app.exec_()
    print(window.points["manual"])

if __name__ == '__main__':
    main()