# importing libraries
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys
# parent_dir = sys.path[0][:sys.path[0].rfind('/', 0, sys.path[0].rfind('/'))] + '/pirl'
# sys.path.insert(0, parent_dir)

from number_widget import NumberWidget
  
class CalibrationScreen(QWidget):
    def __init__(self):
        super().__init__()
        # setting title
        self.setWindowTitle("Autocalibration screen")
        # opening window in maximized size
        self.showFullScreen()
        self.setStyleSheet('background-color: white;')
        # showing all the widgets
        self.calibration_screen = 0
        self.screen_count = QApplication.desktop().screenCount()
        self.widgets = []
        self.hide()
    def set_calibration_screen(self, screen_number):
        self.calibration_screen = screen_number
        self.destroy_widgets()
        self.center(self.calibration_screen)
        self.show()
    def show_screen_number(self):
        for screen in range(self.screen_count):
            print(f"screen: {screen}")
            widget = NumberWidget()
            widget.update_number(screen)
            widget.center(screen)
            widget.label.setMouseCallback(self.set_calibration_screen)
            widget.label.setCallbackParams(screen)
            widget.show()
            self.widgets.append(widget)
    def destroy_widgets(self):
        for screen in range(self.screen_count):
            self.widgets[screen].hide()
            self.widgets[screen].deleteLater()
    def set_color(self, color):
        self.setStyleSheet(f'background-color: {color};')
    def center(self,screen_number):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().screenGeometry(screen_number).center()
        # Offset the window a bit higher
        centerPoint.setY(centerPoint.y())
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

def main():
    # # create pyqt5 app
    App = QApplication(sys.argv)
    # # create the instance of our Window
    window = CalibrationScreen()
    window.show_screen_number()
    sys.exit(App.exec())

if __name__ == '__main__':
    main()