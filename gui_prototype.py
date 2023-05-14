import sys
import io
import subprocess
import os
import numpy as np

from handrec import HandThread
# from pynput.keyboard import Key, Listener
from PyQt5.QtCore import Qt, QThread, QRect, pyqtSignal, pyqtSlot, QBuffer, QPoint
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                             QPushButton, QHBoxLayout, QWidget, QDesktopWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QFont, QPainter, QColor, QPen
from fitz import *

from threading import Thread

from pyqt.select_window import Ui_Form
from utils.autocalibrate import Autocalibration
from utils.cv_wrapper import convert_image


class VirtualCursor(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pressed = False
        self.x = 20
        self.y = 20
        self.has_been_draw = False
        self.setMouseTracking(True)
        self.setCursor(Qt.BlankCursor)
        self._position = QPoint(self.x, self.y)
        self._opacity = 0.4
        self.last_x = None
        self.last_y = None

    def resizeEvent(self, event):
        # When the window is resized, resize the pixmap to the new window size

        pixmap = self.pixmap().scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def paintEvent(self, event):
        # Call the base class implementation to paint the background
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor(255, 255, 0, int(self._opacity * 255)))
        painter.drawEllipse(self._position, 5, 5)
        painter.end()

    def drawEvent(self):
        print("hello")
        painter = QPainter(self.pixmap())

        pen = QPen()
        pen.setColor(QColor("red"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(self.last_x, self.last_y, self.x, self.y)
        painter.end()

        self.last_x = self.x
        self.last_y = self.y
        self.has_been_draw = True

    @pyqtSlot(tuple)
    def mouseMove(self, e):
        self._position = QPoint(e[0], e[1])
        if self.pressed == True:
            self.has_been_draw = True
            if self.last_x is None:  # First event.
                self.last_x = e[0]
                self.last_y = e[1]

            painter = QPainter(self.pixmap())
            pen = QPen()
            pen.setWidth(5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawLine(self.last_x, self.last_y, e[0], e[1])
            painter.end()
            self.update()

            self.last_x = e[0]
            self.last_y = e[1]
        self.update()

    @pyqtSlot(bool)
    def mouseClick(self, clicked):
        if clicked:
            self.pressed = True
        else:
            self.pressed = False
            self.last_x = None
            self.last_y = None

    # def mouseMoveEvent(self, e):

    #     self._position = e.pos()
    #     if self.pressed == True:
    #         self.has_been_draw = True
    #         if self.last_x is None:  # First event.
    #             self.last_x = e.x()
    #             self.last_y = e.y()

    #         painter = QPainter(self.pixmap())
    #         pen = QPen()
    #         pen.setWidth(5)
    #         pen.setCapStyle(Qt.RoundCap)
    #         painter.setPen(pen)
    #         painter.setRenderHint(QPainter.Antialiasing)
    #         painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
    #         painter.end()
    #         self.update()

    #         self.last_x = e.x()
    #         self.last_y = e.y()
    #     self.update()

    # def mousePressEvent(self, e):
    #     self.pressed = True
    #     # print("pressed")

    # def mouseReleaseEvent(self, e):
    #     # print("released")
    #     self.pressed = False
    #     self.last_x = None
    #     self.last_y = None


class pdf_window(QMainWindow):
    def __init__(self, document):
        super().__init__()
        self.has_been_draw = False
        self.setWindowTitle("My App")
        self.doc = document
        self.pno = 0
        self.edited_pdf = fitz.open(document)

        pixmap = self.get_pix_page(self.doc)

        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignCenter)
        layout2 = QHBoxLayout()
        layout2.setAlignment(Qt.AlignCenter)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.label = self.making_canvas(pixmap)

        self.layout.addWidget(self.label)
        next = QPushButton("next")
        next.clicked.connect(self.next)
        previous = QPushButton("previous")
        previous.clicked.connect(self.previous)
        blank = QPushButton("blank")
        blank.clicked.connect(self.blank_page)
        layout2.addWidget(next)
        layout2.addWidget(previous)
        layout2.addWidget(blank)
        self.layout.addLayout(layout2)
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def blank_page(self):
        self.canvas = QPixmap(self.label.width(), self.label.height())
        self.canvas.fill(Qt.white)
        self.label = self.making_canvas(self.canvas)
        self.layout.takeAt(0)
        self.layout.insertWidget(0, self.label)

    def get_pix_page(self, doc):
        page = doc[self.pno]
        zoom = 2    # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pixmap_image = page.get_pixmap(matrix=mat)
        bytes = QImage.fromData(pixmap_image.tobytes())
        pixmap = QPixmap.fromImage(bytes)

        return pixmap

    def making_canvas(self, pixmap):
        label = VirtualCursor()
        self.setMinimumSize(1000, 900)
        self.pixmap = pixmap.scaled(
            self.size(), aspectRatioMode=Qt.KeepAspectRatio)
        label.setPixmap(self.pixmap)
        return label

    # def resizeEvent(self, event):
    #     # When the window is resized, resize the pixmap to the new window size
    #     self.pixmap = self.pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio)
    #     self.label.setPixmap(self.pixmap)

    def write_next(self):

        image = self.temp_pix.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, 'PNG')  # Save the QImage as a PNG to the buffer
        image_bytes = buffer.data()
        file = io.BytesIO(image_bytes)
        new_doc = fitz.open(stream=file.read(), filetype="png")
        pdfBytes = new_doc.convert_to_pdf()
        new_doc.close()
        copy = fitz.open("pdf", pdfBytes)
        page = self.edited_pdf.new_page(
            self.pno - 1, self.doc[0].rect.width, self.doc[0].rect.height)
        page.show_pdf_page(self.doc[0].rect, copy, 0)
        pages = list(p for p in range(
            self.edited_pdf.page_count) if p != self.pno)
        self.edited_pdf.select(pages)
        self.edited_pdf.save("see.pdf")

    def write_previous(self):

        image = self.temp_pix.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        image.save(buffer, 'PNG')  # Save the QImage as a PNG to the buffer
        image_bytes = buffer.data()
        file = io.BytesIO(image_bytes)
        new_doc = fitz.open(stream=file.read(), filetype="png")
        pdfBytes = new_doc.convert_to_pdf()
        new_doc.close()
        copy = fitz.open("pdf", pdfBytes)
        page = self.edited_pdf.new_page(
            self.pno + 1, self.doc[0].rect.width, self.doc[0].rect.height)
        page.show_pdf_page(self.doc[0].rect, copy, 0)
        pages = list(p for p in range(
            self.edited_pdf.page_count) if p != self.pno + 2)
        self.edited_pdf.select(pages)
        self.edited_pdf.save("see.pdf")

    def move(self, str):
        print("move")
        self.label.mouseMoveEvent(str)
        self.update()
        self.has_been_draw = True

    def paintEvent(self, e):
        if self.label.pressed == True:
            self.has_been_draw = True

    def next(self):
        if self.pno == self.doc.__len__() - 1:
            self.pno = 0
        else:
            self.pno += 1
        if self.label.has_been_draw == True:
            self.temp_pix = self.label.pixmap()
            pix = self.get_pix_page(self.doc)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)

            self.thread = MyThread(self, self.write_next)
            self.thread.started.connect(lambda: print("writing started!"))
            self.thread.finished.connect(lambda: print("writing finished!"))
            self.thread.start()

            self.has_been_draw == False
        else:
            pix = self.get_pix_page(self.edited_pdf)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)

    def previous(self):
        if self.pno == 0:
            self.pno = self.doc.__len__() - 1
        else:
            self.pno -= 1
        if self.label.has_been_draw == True:
            self.temp_pix = self.label.pixmap()
            pix = self.get_pix_page(self.edited_pdf)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)

            self.thread = MyThread(self, self.write_previous)
            self.thread.started.connect(lambda: print("writing started!"))
            self.thread.finished.connect(lambda: print("writing finished!"))
            self.thread.start()

            self.has_been_draw == False
        else:
            pix = self.get_pix_page(self.edited_pdf)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)


class MyThread(QThread):
    started = pyqtSignal()  # Custom signal for starting the thread
    finished = pyqtSignal()  # Custom signal for finishing the thread

    def __init__(self, pdf_window, func, parent=None):
        super().__init__(parent)
        self.func = func
        self.window = pdf_window

    def run(self):
        self.started.emit()
        self.func()
        self.finished.emit()
        print(f"end of {self.func()}")


class HandWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hand_thread = HandThread()
        self.image = QLabel(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('hand_window')
        self.resize(640, 480)  # default size
        
        # Set the central widget of the window to the image label
        self.setCentralWidget(self.image)

        # Set the minimum size of the window to the initial size of the label
        self.setMinimumSize(self.image.sizeHint())

    def begin(self):
        self.hand_thread.change_pixmap_signal.connect(self.update_image)
        self.hand_thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = convert_image(cv_img, self.image.size().width(), self.image.size().height())
        self.image.setPixmap(qt_img)
        
    def resizeEvent(self, event):
        # Resize the label to match the size of the window
        self.image.resize(self.width(), self.height())
        event.accept()

    def set_homography_points(self, points):
        self.hand_thread.set_homography_points(points)

    def closeEvent(self, event):
        # Call the hand_thread.stop() function before closing the window
        self.hand_thread.stop()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self,):
        super().__init__()

        self.autocalibrator = Autocalibration()
        self.hand_window = HandWindow()
        self.hand_window.hide()

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

        self.new_pdf_button = QPushButton("Empty File")
        self.new_pdf_button.setFont(font)
        self.new_pdf_button.clicked.connect(self.create_pdf)

        label = QLabel("welcome to PIRL")
        label.setFont(QFont("Arial", 20))
        layout.addWidget(label)
        layout.addWidget(self.button)
        layout.addWidget(self.calibration_button)
        layout.addWidget(self.new_pdf_button)

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
            self.w = pdf_window(doc)
            screen = QDesktopWidget().screenGeometry(0)
            self.w.setGeometry(QRect(screen))
            self.hand_window.hand_thread.coor_signal.connect(self.w.label.mouseMove)
            self.hand_window.hand_thread.click_signal.connect(self.w.label.mouseClick)
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
        self.w = pdf_window(doc)
        screen = QDesktopWidget().screenGeometry(0)
        self.w.setGeometry(QRect(screen))
        self.w.show()

    def calibrate_screen(self):
        self.autocalibrator.autocalibrate()
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


def main():
    # print(qt5_vars)
    # you have to delete qt5 variables before using qt5, i don't understand why.
    # autocalibrator = Autocalibration()
    # autocalibrator.delete_qt_vars()
    app = QApplication([])
    window = MainWindow()
    window.autocalibrator.delete_qt_vars()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
