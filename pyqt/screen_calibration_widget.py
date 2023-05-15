# importing libraries
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import numpy as np
from time import sleep
import sys, os
sys.path.append(os.path.abspath('.'))
from pyqt.number_widget import NumberWidget
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
        widget = self.ButtonWidget(lambda: self.set_calibration_screen(0),lambda: self.set_calibration_screen(1))
        widget.show()
        self.widgets.append(widget)

    def destroy_widgets(self):
        for widget in self.widgets:
            widget.hide()
            widget.deleteLater()

    def set_color(self, color):
        self.setStyleSheet(f'background-color: {color};')
    def center(self,screen_number):
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