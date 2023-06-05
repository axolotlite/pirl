# from pynput.keyboard import Key, Listener
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                             QPushButton, QWidget, QDesktopWidget, QFileDialog)
from PyQt5.QtGui import QFont
from fitz import fitz

from pyqt.select_window import Ui_Form
from pyqt.windows import HandWindow, CameraSelectWindow, ScreenSelectWindow, PDFWindow
from utils.autocalibrate import Autocalibration
from utils.cv_wrapper import convert_image

from cfg import CFG


class MainWindow(QMainWindow):
    def __init__(self,):
        super().__init__()

        self.autocalibrator = Autocalibration()
        self.hand_window = HandWindow()
        self.hand_window.hide()

        self.csw = CameraSelectWindow()
        self.ssw = ScreenSelectWindow()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("choose a file")
        font = self.button.font()
        font.setPointSize(10)
        self.button.setFont(font)
        self.setFixedSize(800, 500)

        self.calibration_button = QPushButton("Calibrate")
        self.calibration_button.setFont(font)
        self.calibration_button.clicked.connect(self.calibrate_screen)

        self.result_button = QPushButton("Calibration Result")
        self.result_button.setFont(font)
        self.result_button.clicked.connect(self.show_results)

        self.new_pdf_button = QPushButton("Empty File")
        self.new_pdf_button.setFont(font)
        self.new_pdf_button.clicked.connect(self.create_pdf)

        self.set_cam_button = QPushButton("Choose camera")
        self.set_cam_button.setFont(font)
        self.set_cam_button.clicked.connect(self.set_cam)

        self.set_screen_button = QPushButton("Choose main screen")
        self.set_screen_button.setFont(font)
        self.set_screen_button.clicked.connect(self.set_screen)

        label = QLabel("welcome to PIRL")
        label.setFont(QFont("Arial", 20))
        layout.addWidget(label)
        layout.addWidget(self.button)
        layout.addWidget(self.calibration_button)
        layout.addWidget(self.result_button)
        layout.addWidget(self.new_pdf_button)
        layout.addWidget(self.set_cam_button)
        layout.addWidget(self.set_screen_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.button.clicked.connect(self.choose_file)
        self.setCentralWidget(widget)

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "PDF Files (*.pdf)", options=options)
        if file_name:
            print(f"Selected file: {file_name}")
            doc = fitz.open(file_name)
            self.w = PDFWindow(doc)
            self.w.set_hand_thread(self.hand_window.hand_thread)
            screen = QDesktopWidget().screenGeometry(CFG.mainScreen)
            self.w.setGeometry(QRect(screen))
            # here you pass just the first mouse move of the first page when we need to pass the mouseMove the page we are in since we always do another cursor when new page
            self.hand_window.hand_thread.coor_signal.connect(
                self.w.label.mouseMove)
            self.hand_window.hand_thread.click_signal.connect(
                self.w.label.mouseClick)
            self.show()
            self.w.showMaximized()
            self.hide()

    def create_pdf(self):
        print("hello world")
        doc = fitz.open()
        doc._newPage()
        doc.write()
        doc.save("tmp.pdf")
        doc = fitz.open("tmp.pdf")
        self.w = PDFWindow(doc)
        screen = QDesktopWidget().screenGeometry(0)
        self.w.setGeometry(QRect(screen))
        self.w.show()

    def show_results(self):
        # self.autocalibrator.black_screen = self.autocalibration_thread.images[0]
        # self.autocalibrator.white_screen = self.autocalibration_thread.images[1]
        # self.autocalibrator.autocalibrate()

        image = self.autocalibrator.get_masked_image("diff_mask")
        image2 = self.autocalibrator.get_masked_image("boundaries_mask")

        self.w = QWidget()
        ui = Ui_Form()
        ui.setupUi(self.w)

        def set_point(mask_type):
            if(mask_type == "manual"):
                self.autocalibrator.fallback_calibration()
                self.autocalibrator.set_points(mask_type)
            else:
                self.autocalibrator.set_points(mask_type)
            self.hand_window.set_homography_points(
                self.autocalibrator.default_points)
            self.hand_window.begin()
            self.hand_window.show()
            self.w.deleteLater()

        ui.first_image.setPixmap(convert_image(image))
        ui.first_image.setMouseCallback(lambda: set_point("diff_mask"))
        ui.second_image.setPixmap(convert_image(image2))
        ui.second_image.setMouseCallback(lambda: set_point("boundaries_mask"))
        ui.pushButton.clicked.connect(lambda: set_point("manual"))
        self.w.show()

    def calibrate_screen(self):
        self.autocalibrator.create_widget()
        QTimer.singleShot(500, self.autocalibrator.start_calibration)

    def set_cam(self):
        self.csw.start()

    def set_screen(self):
        self.ssw.show()

    # def keyboard_pressing(self):
    #     with Listener( on_press=self.on_press, on_release= None) as listener:
    #         listener.join()
    #     self.w.close()

    # def closeEvent(self, event):
    #     self.close()

# print(autocalibrate())


def main():
    # print(qt5_vars)
    # you have to delete qt5 variables before using qt5, i don't understand why.
    # autocalibrator = Autocalibration()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
