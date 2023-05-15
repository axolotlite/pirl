# importing libraries
from pyqt.number_widget import NumberWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
from time import sleep
import cv2
import sys
import os
sys.path.append(os.path.abspath('.'))


class CalibrationScreen(QWidget):
    def __init__(self):
        super().__init__()
        # setting title
        self.setWindowTitle("Autocalibration screen")
        # opening window in maximized size
        self.setStyleSheet('background-color: black;')
        # showing all the widgets
        self.calibration_screen = None
        # self.screen_count = QApplication.desktop().screenCount()
        self.widgets = []

    def set_calibration_screen(self, screen_number):
        self.calibration_screen = screen_number
        self.destroy_widgets()
        self.center(self.calibration_screen)
        screen = QDesktopWidget().screenGeometry(screen_number)
        self.setWindowState(Qt.WindowMaximized)
        self.setMaximumSize(screen.size())
        self.showFullScreen()

    def select_screen(self):
        widget = self.ButtonWidget(lambda: self.set_calibration_screen(
            0), lambda: self.set_calibration_screen(1))
        widget.show()
        self.widgets.append(widget)

    def destroy_widgets(self):
        for widget in self.widgets:
            widget.hide()
            widget.deleteLater()

    def set_color(self, color):
        self.setStyleSheet(f'background-color: {color};')

    def center(self, screen_number):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().screenGeometry(screen_number).center()
        # Offset the window a bit higher
        centerPoint.setY(centerPoint.y())
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    class ButtonWidget(QWidget):
        def __init__(self, func1, func2):
            super().__init__()

            # Create two buttons
            self.button1 = QPushButton('screen 0', self)
            self.button2 = QPushButton('screen 1', self)
            self.button1.clicked.connect(func1)
            self.button2.clicked.connect(func2)
            # Create a layout for the widget and add the buttons to it
            layout = QVBoxLayout()
            layout.addWidget(self.button1)
            layout.addWidget(self.button2)

            # Set the layout for the widget
            self.setLayout(layout)


class ManualScreen(QMainWindow):
    def __init__(self, camIdx=0, screen=None):
        super().__init__()
        self.camIdx = camIdx
        self.screen = screen
        self.count = 0
        self.points = {"manual": []}
        self.breakflag = False

        self.cap = cv2.VideoCapture(self.camIdx)
        self.cap_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.label = QLabel(self)
        self.label.setFixedSize(self.cap_width, self.cap_height)
        self.setCentralWidget(self.label)

        self.setWindowTitle("Calibration")
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)
        # self.move(int(self.screen.width() * 1.3), int(self.screen.height() * 0.3))
        # QTimer.singleShot(500, self.fallback_calibration)
        self.show()
        self.fallback_calibration()

    def mousePressEvent(self, event):
        if self.count < 4:
            x, y = event.pos().x(), event.pos().y()
            if self.count == 0:
                self.points["manual"].append([x, y])
            if self.count < 3:
                self.points["manual"].append([x, y])
            self.points["manual"][self.count] = [x, y]
            self.count += 1

    def mouseMoveEvent(self, event):
        if not (self.count == 0 or self.count == 4):
            x, y = event.pos().x(), event.pos().y()
            self.points["manual"][self.count] = [x, y]

    def fallback_calibration(self):
        waitTime = 50
        # reset points
        self.count = 0
        self.points["manual"] = []

        while (self.cap.isOpened()):

            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            if self.count == 4 or self.breakflag:
                break

            for i in range(len(self.points["manual"])):
                if (i + 1 == len(self.points["manual"])):
                    cv2.line(frame, self.points["manual"][i],
                             self.points["manual"][0], (187, 87, 231), 2)
                else:
                    cv2.line(
                        frame, self.points["manual"][i], self.points["manual"][i+1], (187, 87, 231), 2)

            pixmap = QPixmap.fromImage(
                QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_BGR888))
            self.label.setPixmap(pixmap)

            QApplication.processEvents()

        self.cap.release()
        print(self.points["manual"])
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_B:
            self.breakflag = True
        elif event.key() == Qt.Key_R:
            self.count = 0
            self.points["manual"] = []
        event.accept()

    def closeEvent(self, event: QCloseEvent):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


def main():
    # # create pyqt5 app
    App = QApplication(sys.argv)
    # # create the instance of our Window
    window = CalibrationScreen()
    # window.thread.start()
    # window.capture_screen()
    window.select_screen()
    sys.exit(App.exec())


if __name__ == '__main__':
    main()
