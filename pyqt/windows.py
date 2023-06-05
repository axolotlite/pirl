import sys
import os
sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
from cfg import CFG
from utils.cv_wrapper import convert_image
from recognition.hand import HandThread
from fitz import fitz
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtMultimedia import QCamera, QCameraInfo, QCameraImageCapture
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QFont, QPainter, QColor, QPen
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                             QPushButton, QHBoxLayout, QWidget, QDesktopWidget, QFileDialog, QStatusBar, QToolBar, QAction, QComboBox, QErrorMessage)
from PyQt5.QtCore import Qt, QThread, QRect, pyqtSignal, pyqtSlot, QBuffer, QPoint, QTimer
import numpy as np
import io


class VirtualCursor(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pressed = False
        self.x = 0
        self.y = 0
        self.has_been_draw = False
        self.setMouseTracking(True)
        self.setCursor(Qt.BlankCursor)
        self._position = QPoint(self.x, self.y)
        self._opacity = 0.4
        self.last_x = None
        self.last_y = None
        self.global_x = self.mapToGlobal(QPoint(0, 0)).x()

    def setCoordinates(self):
        self.global_x_min = self.mapToGlobal(QPoint(0, 0)).x()
        self.global_y_min = self.mapToGlobal(QPoint(0, 0)).y()
        self.global_x_max = self.mapToGlobal(QPoint(self.width(), 0)).x()
        self.global_y_max = self.mapToGlobal(QPoint(0, self.height())).y()
        # Adjust coordinates if not primary screen
        if CFG.mainScreen:
            self.global_x_min -= CFG.monitors[CFG.mainScreen].x
            self.global_y_min -= CFG.monitors[CFG.mainScreen].y
            self.global_x_max -= CFG.monitors[CFG.mainScreen].x
            self.global_y_max -= CFG.monitors[CFG.mainScreen].y
        # print(
        #     f"Self xmin = {self.global_x_min}, self ymin = {self.global_y_min}")
        # print(
        #     f"Self xmax = {self.global_x_max}, self ymax = {self.global_y_max}")

    def normalizeCoordinates(self, e):
        if e[0] >= self.global_x_min and e[0] <= self.global_x_max and e[1] >= self.global_y_min and e[1] <= self.global_y_max:
            e = (e[0] - self.global_x_min, e[1] - self.global_y_min)
            return True, e
        return False, e

    # def moveEvent(self, event) -> None:
    #     self.setCoordinates()
    #     return super().moveEvent(event)

    def resizeEvent(self, event) -> None:
        # When the window is resized, resize the pixmap to the new window size
        # print(f" resize label {self.mapToGlobal(QPoint(0,0))}")
        self.setCoordinates()
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

    @pyqtSlot(tuple)
    def mouseMove(self, e):
        success, e = self.normalizeCoordinates(e)
        if success:
            self._position = QPoint(e[0], e[1])
            # print(f"in move {self._position}")
            # print(self.mapToGlobal(self._position))
            # print(self.mapFromGlobal(self.mapToGlobal(self._position)))
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


class PDFWindow(QMainWindow):
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

    def set_hand_thread(self, hand_thread):
        print("let's gooo")
        self.hand_thread = hand_thread

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

    def resizeEvent(self, event):
        # When the window is resized, resize the pixmap to the new window size
        # print(F"resize pdf {self.mapToGlobal(QPoint(0,0))}")
        self.pixmap = self.pixmap.scaled(
            self.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

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
            self.hand_thread.coor_signal.connect(self.label.mouseMove)
            self.hand_thread.click_signal.connect(self.label.mouseClick)
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
            self.hand_thread.coor_signal.connect(self.label.mouseMove)
            self.hand_thread.click_signal.connect(self.label.mouseClick)
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
            self.hand_thread.coor_signal.connect(self.label.mouseMove)
            self.hand_thread.click_signal.connect(self.label.mouseClick)
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
            self.hand_thread.coor_signal.connect(self.label.mouseMove)
            self.hand_thread.click_signal.connect(self.label.mouseClick)
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
        qt_img = convert_image(
            cv_img, self.image.size().width(), self.image.size().height())
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


class CameraSelectWindow(QMainWindow):

    # constructor
    def __init__(self):
        super().__init__()

        # setting geometry
        self.setGeometry(100, 100,
                         800, 600)

        # setting style sheet
        self.setStyleSheet("background : lightgrey;")

        # getting available cameras
        self.available_cameras = QCameraInfo.availableCameras()

        # if no camera found
        if not self.available_cameras:
            # exit the code
            sys.exit()

        # creating a QCameraViewfinder object
        self.viewfinder = QCameraViewfinder()

        # creating a combo box for selecting camera
        self.camera_selector = QComboBox()

        # adding status tip to it
        self.camera_selector.setStatusTip("Choose desired camera")

        # adding tool tip to it
        self.camera_selector.setToolTip("Select Camera")
        self.camera_selector.setToolTipDuration(2500)

        # adding items to the combo box
        self.camera_selector.addItems([camera.description()
                                       for camera in self.available_cameras])

        # adding action to the combo box
        # calling the select camera method
        self.camera_selector.currentIndexChanged.connect(self.select_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.viewfinder)
        layout.addWidget(self.camera_selector)

        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)

        # setting window title
        self.setWindowTitle("PyQt5 Cam")

    def start(self):
        # showing this viewfinder
        self.viewfinder.show()
        # Set the default camera.
        self.camera_selector.setCurrentIndex(CFG.camIdx)
        self.select_camera(CFG.camIdx)
        self.show()

    # method to select camera
    def select_camera(self, i):

        # getting the selected camera
        self.camera = QCamera(self.available_cameras[i])

        # setting view finder to the camera
        self.camera.setViewfinder(self.viewfinder)

        # setting capture mode to the camera
        self.camera.setCaptureMode(QCamera.CaptureStillImage)

        # if any error occur show the alert
        self.camera.error.connect(
            lambda: self.alert(self.camera.errorString()))

        # start the camera
        self.camera.start()

        # creating a QCameraImageCapture object
        self.capture = QCameraImageCapture(self.camera)

        # showing alert if error occur
        self.capture.error.connect(lambda error_msg, error,
                                   msg: self.alert(msg))

        # when image captured showing message
        self.capture.imageCaptured.connect(lambda d,
                                           i: self.status.showMessage("Image captured : "
                                                                      + str(self.save_seq)))

        # getting current camera name
        self.current_camera_name = self.available_cameras[i].description()

        # initial save sequence
        self.save_seq = 0

        CFG.camIdx = i

    # method for alerts
    def alert(self, msg):

        # error message
        error = QErrorMessage(self)

        # setting text to the error message
        error.showMessage(msg)


class ScreenSelectWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.screens = QApplication.instance().screens()

        left_screen = QLabel()
        right_screen = QLabel()

        # Resize the screen captures so they fit inside the window.
        left_screen.setScaledContents(True)
        right_screen.setScaledContents(True)

        left_screen.setPixmap(self.screens[0].grabWindow(0))

        if len(self.screens) == 2:
            right_screen.setPixmap(self.screens[1].grabWindow(0))
        else:
            right_screen.setText("No other monitor detected")

        layout = QHBoxLayout()
        layout.addWidget(left_screen)
        layout.addWidget(right_screen)

        self.setLayout(layout)

        self.left_screen = left_screen
        self.right_screen = right_screen

        self.left_screen.mousePressEvent = self.on_left_screen_clicked
        self.right_screen.mousePressEvent = self.on_right_screen_clicked

        # Resize the window to a third of the size of the screen.
        self.resize(self.screens[0].size() / 2)

        # Resize the labels to fit the size of the window.
        self.left_screen.setFixedSize(self.width() / 2, self.height())
        self.right_screen.setFixedSize(self.width() / 2, self.height())

    def on_left_screen_clicked(self, event):
        CFG.mainScreen = 0
        self.hide()

    def on_right_screen_clicked(self, event):
        CFG.mainScreen = 1
        self.hide()
