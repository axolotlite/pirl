import sys, os

sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
from fitz import fitz
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from pyqt.windows import *


def main():
    app = QApplication([])
    doc = fitz.open("test.pdf")
    w = PDFWindow()
    w.set_doc(doc)
    hand_window = HandWindow()
    w.set_hand_thread(hand_window.hand_thread)
    screen = QDesktopWidget().screenGeometry(0)
    w.setGeometry(QRect(screen))
    hand_window.hand_thread.coor_signal.connect(w.cursor.mouseMove)
    hand_window.hand_thread.click_signal.connect(w.cursor.mouseClick)
    hand_window.set_homography_points([[118, 43], [118, 444], [509, 421], [483, 46]])
    hand_window.begin()
    hand_window.show()
    w.showMaximized()
    app.exec_()


if __name__ == "__main__":
    main()
