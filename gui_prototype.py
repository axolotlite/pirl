# from pynput.keyboard import Key, Listener
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QDesktopWidget,
    QFileDialog,
)
from PyQt5.QtGui import QFont, QPalette, QColor
from fitz import fitz

from pyqt.select_window import Ui_Form
from pyqt.windows import HandWindow, CameraSelectWindow, ScreenSelectWindow, PDFWindow
from pyqt.classroom_window import ClassroomCompanion
from utils.autocalibrate import Autocalibration
from utils.cv_wrapper import convert_image

from cfg import CFG


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.autocalibrator = Autocalibration()
        self.pdfw = PDFWindow()
        self.handw = HandWindow()
        self.csw = CameraSelectWindow()
        self.ssw = ScreenSelectWindow()
        self.classroomCompanion = ClassroomCompanion()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setFixedSize(800, 500)

        font_btn = QFont("Arial", 10)
        font_ttl = QFont("Arial", 20)

        title = QLabel("Welcome to PIRL")
        title.setFont(font_ttl)

        self.button = QPushButton("Choose a file")
        self.button.setFont(font_btn)
        self.button.clicked.connect(self.choose_file)

        self.new_pdf_button = QPushButton("Empty File")
        self.new_pdf_button.setFont(font_btn)
        self.new_pdf_button.clicked.connect(self.create_pdf)

        self.calibration_button = QPushButton("Automatic Calibration")
        self.calibration_button.setFont(font_btn)
        self.calibration_button.clicked.connect(self.calibrate_screen)

        self.result_button = QPushButton("Calibration Result")
        self.result_button.setFont(font_btn)
        self.result_button.clicked.connect(self.show_results)
        self.manual_button = QPushButton("Manual Calibration")
        self.manual_button.setFont(font_btn)
        self.manual_button.clicked.connect(self.manual_calibration)

        self.set_cam_button = QPushButton("Choose camera")
        self.set_cam_button.setFont(font_btn)
        self.set_cam_button.clicked.connect(self.set_cam)

        self.set_screen_button = QPushButton("Choose main screen")
        self.set_screen_button.setFont(font_btn)
        self.set_screen_button.clicked.connect(self.set_screen)

        self.classroom_companion_button = QPushButton("Classroom Companion")
        self.classroom_companion_button.setFont(font_btn)
        self.classroom_companion_button.clicked.connect(self.classroomCompanion.show)

        # self.classroomCompanion

        layout.addWidget(title)
        layout.addWidget(self.button)
        layout.addWidget(self.new_pdf_button)
        layout.addWidget(self.calibration_button)
        layout.addWidget(self.result_button)
        layout.addWidget(self.manual_button)
        layout.addWidget(self.set_cam_button)
        layout.addWidget(self.set_screen_button)
        layout.addWidget(self.classroom_companion_button)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "PDF Files (*.pdf)", options=options
        )
        if file_name:
            print(f"Selected file: {file_name}")
            doc = fitz.open(file_name)
            self.pdfw.set_doc(doc)
            self.pdfw.set_hand_thread(self.handw.hand_thread)
            self.pdfw.connect_hand_thread()
            screen = QDesktopWidget().screenGeometry(CFG.mainScreen)
            self.pdfw.setGeometry(QRect(screen))
            self.pdfw.showFullScreen()
            

    def create_pdf(self):
        doc = fitz.open()
        doc._newPage()
        doc.write()
        doc.save("tmp.pdf")
        doc = fitz.open("tmp.pdf")
        self.pdfw.set_doc(doc)
        self.pdfw.set_hand_thread(self.handw.hand_thread)
        self.pdfw.connect_hand_thread()
        screen = QDesktopWidget().screenGeometry(CFG.mainScreen)
        self.pdfw.setGeometry(QRect(screen))
        self.pdfw.showFullScreen()
        self.hide()

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
            if mask_type == "manual":
                self.autocalibrator.fallback_calibration()
                self.autocalibrator.set_points(mask_type)
            else:
                self.autocalibrator.set_points(mask_type)
            self.handw.set_homography_points(self.autocalibrator.default_points)
            self.handw.begin()
            self.handw.show()
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

    def manual_calibration(self):
        self.autocalibrator.fallback_calibration()
        self.autocalibrator.set_points("manual")
        self.handw.set_homography_points(self.autocalibrator.default_points)
        self.handw.begin()
        self.handw.show()

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
    # Force the style to be the same on all OSs:
    # app.setStyle("Fusion")
    # Now use a palette to switch to dark colors:

    window = MainWindow()
    app.lastWindowClosed.connect(window.classroomCompanion.api.stop)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
    # print("hello")
