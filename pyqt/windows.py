import sys
import os

sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
from cfg import CFG
from utils.cv_wrapper import convert_image
from recognition.hand import HandThread
from fitz import fitz
from time import sleep
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtMultimedia import QCamera, QCameraInfo, QCameraImageCapture
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QFont, QPainter, QColor, QPen, QMouseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QStackedLayout,
    QPushButton,
    QHBoxLayout,
    QWidget,
    QDesktopWidget,
    QFileDialog,
    QStatusBar,
    QToolBar,
    QAction,
    QComboBox,
    QErrorMessage,
)
from PyQt5.QtCore import (
    Qt,
    QThread,
    QRect,
    pyqtSignal,
    pyqtSlot,
    QBuffer,
    QPoint,
    QTimer,
)
import numpy as np
import io


class VirtualCursor(QLabel):
    def __init__(self,pixmap ,parent=None):
        super().__init__(parent)
        print("virtual cusror has been initiated")
        self.setPixmap(pixmap)
        self.pressed = False
        self.setFixedSize(pixmap.size())
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

    def resizeEvent(self, event) -> None:
        # When the window is resized, resize the pixmap to the new window size
        print(f"in virtual mask size = {self.size()}")
        print(f"in virtual mask  pixmap size = {self.pixmap().size()}")
        pixmap = self.pixmap().scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(pixmap)
        print(self.mapToGlobal(self.pos()))



class DrawingMask(QLabel):
    font_size = 5
    font_color = 'black'
    def __init__(self,virtual_cursor: VirtualCursor ,parent=None):
        super().__init__(parent)
        print("Drawing Mask has been initiated")
        pixmap = QPixmap(virtual_cursor.pixmap().size())
        pixmap.fill(Qt.transparent)
        self.setPixmap(pixmap)
        self.setStyleSheet("background-color: transparent;")
        self.setFixedSize(pixmap.size())
        self.pressed = False
        self.x = 0
        self.y = 0      
        self.has_been_draw = False
        self.setMouseTracking(True)
        self.setCursor(Qt.BlankCursor)
        self._position = QPoint(self.x,self.y)
        self._opacity = 0.4
        self.erase = False
        self.last_x = None
        self.last_y = None
    
    def resizeEvent(self, event):
        # When the window is resized, resize the pixmap to the new window size
        
        print(f"in drawing mask size = {self.size()}")
        print(f"in drawing mask  pixmap size = {self.pixmap().size()}")
        pixmap = self.pixmap().scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(pixmap)
        print(self.mapToGlobal(self.pos()))

    
    def paintEvent(self, event):
        # Call the base class implementation to paint the background
        super().paintEvent(event)
        # print(self.mapToGlobal(self.pos()))
        # print("paint event !")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(255,255,0,int(self._opacity * 255)))
        painter.drawEllipse(self._position, self.font_size, self.font_size)
        painter.end()
        


    @pyqtSlot(tuple)
    def mouseMove(self, e):
        # print("moving")
        # success, e = self.normalizeCoordinates(e)
        self._position = self.mapFromParent(QPoint(e[0], e[1]))
        e = (self._position.x(), self._position.y())

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
            if self.font_color == 'transparent':
                pen.setBrush(QColor(Qt.transparent))
                painter.setCompositionMode(QPainter.CompositionMode_Source)
            else:
                pen.setBrush(QColor(self.font_color))
            pen.setWidth(self.font_size)
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

        


class Label(QLabel):
    def __init__(self,virtual: DrawingMask ,widgets: list[QPushButton],parent=None):
        super().__init__(parent)
        self.widgets_list =  widgets # so we can know all places of all widgets
        self.vir = virtual
        self.x = 0
        self.y = 0      
        self.has_been_draw = False
        self.setMouseTracking(True)
        self.setCursor(Qt.BlankCursor)
        self._position = QPoint(self.x,self.y)
        self.outside = True
        self.pressed = False
    def paintEvent(self, event):
        # Call the base class implementation to paint the background
        super().paintEvent(event)
        # print(self.mapToGlobal(self.pos()))

        if self.outside:
            
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            painter.setBrush(QColor(255,0,0,int(0.5* 255)))
            painter.drawEllipse(self._position, 5, 5)
            painter.end()
        
    def mouseClick(self, clicked):
        if clicked:
            self.pressed = True
        else:
            self.pressed = False


    def mouseMove(self, e : tuple):
        clicked = False
        # print(f"{self.mapToGlobal(e.pos())} pointer")
        if(( e[0] <= self.vir.mapToParent(QPoint(0,0)).x() + self.vir.width()  ) and ( e[1] <= self.vir.mapToParent(QPoint(0,0)).y() + self.vir.height())) and (( e[0] >= self.vir.mapToParent(QPoint(0,0)).x()   ) and ( e[1] >= self.vir.mapToParent(QPoint(0,0)).y() )):
            self.outside = False
            self.vir.mouseClick(self.pressed)
            self.vir.mouseMove(e) 
        else:
            
            self.outside = True
            for widget in self.widgets_list:
                if(( e[0] <= widget.mapToParent(QPoint(0,0)).x() + widget.width()  ) and ( e[1] <= widget.mapToParent(QPoint(0,0)).y() + widget.height())) and (( e[0] >= widget.mapToParent(QPoint(0,0)).x()   ) and ( e[1] >= widget.mapToParent(QPoint(0,0)).y() )):
                    # print("True")
                    
                    if self.pressed:
                        
                        self.pressed = False
                        widget.click()
                        self.stop_hand_thread()
                        clicked = True
                        sleep(0.4)
                        self.start_hand_thread()
                        
            
        self._position = QPoint(e[0], e[1]) if clicked == False else QPoint(0,0)
        
        self.update()
       

    def changeVir(self, new_vir):
        self.vir = new_vir
    def set_hand_thread(self, hand_thread: HandThread):
        self.hand_thread = hand_thread

    def stop_hand_thread(self):
        self.hand_thread._run_flag = False
    def start_hand_thread(self):
        self.hand_thread._run_flag = True
        
class Page():
    def __init__(self,mask: DrawingMask, page_no: int ,blank = False,parent=None):
        self.mask = mask
        self.page_no = page_no
            
class PDFWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.has_been_draw = False
        self.setWindowTitle("My App")
        

    def set_doc(self, document):
        self.container = QWidget()
        stack = QStackedLayout()
        stack.setStackingMode(QStackedLayout.StackAll)
        self.stack_drawing = QStackedLayout()
        self.stack_drawing.setStackingMode(QStackedLayout.StackAll)
        
        self.doc = document
        
        self.pno = 0
        self.page_label_no = QLabel(f"page : {self.pno + 1} of {document.__len__()}")
        self.page_label_no.setFont(QFont('Arial', 15))

        self.edited_pdf = fitz.open(document)
        self.blank = False
        self.setMouseTracking(True)
        self.setCursor(Qt.BlankCursor)
        self.pixmap = self.get_pix_page(self.doc)
        self.main_layout = QHBoxLayout()
        self.container.setLayout(self.main_layout)
        self.main_layout.setSpacing(10)
        self.main_layout.setAlignment(Qt.AlignCenter)
        layout2 = QVBoxLayout()
        layout2.setAlignment(Qt.AlignCenter)
        self.main_layout.setContentsMargins(5,5,5,5)
        
        layout3 = QVBoxLayout()
        layout3.setAlignment(Qt.AlignCenter)
        
        
        self.label = self.making_canvas()
        self.drawing_mask = DrawingMask(self.label)

        # self.label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.font_size_label = QLabel(f"font size : {self.drawing_mask.font_size}")
        self.font_size_label.setFont(QFont('Arial', 15))
        self.font_color_label = QLabel(f"current color : {self.drawing_mask.font_color}")
        self.font_color_label.setFont(QFont('Arial', 15))
        increase_font = QPushButton("increase font")
        decrease_font = QPushButton("decrease font")
        increase_font.clicked.connect(self.increase_font)
        decrease_font.clicked.connect(self.decrease_font)
        
        
        layout3.addWidget(self.font_size_label)
        layout3.addWidget(increase_font)
        layout3.addWidget(decrease_font)
        layout3.addWidget(self.font_color_label)
        next = QPushButton("next")
        next.clicked.connect(self.next)
        previous = QPushButton("previous")
        previous.clicked.connect(self.previous)
        blank = QPushButton("blank")
        blank.clicked.connect(self.blank_page)
        layout2.addWidget(self.page_label_no)
        layout2.addWidget(next)
        layout2.addWidget(previous)
        layout2.addWidget(blank)
        self.widgets = []
        color_layout = self.color_layout()
        layout3.addLayout(color_layout)
        self.widgets.append(next)
        self.widgets.append(previous)
        self.widgets.append(blank)
        self.widgets.append(increase_font)
        self.widgets.append(decrease_font)
        self.main_layout.addLayout(layout2)
        self.main_layout.addLayout(self.stack_drawing)
        self.main_layout.addLayout(layout3)
        widget = QWidget()
        self.biggest_label = Label(self.drawing_mask, self.widgets)

        self.biggest_label.setStyleSheet("background-color: transparent;")
        stack.addWidget(self.container)
        stack.addWidget(self.biggest_label)
        widget.setLayout(stack)
        self.setCentralWidget(widget)
        self.stack_drawing.addWidget(self.drawing_mask)
        self.stack_drawing.addWidget(self.label)
        self.pages = [Page(self.drawing_mask, 0)]
        

    def set_hand_thread(self, hand_thread):
        self.hand_thread = hand_thread
        self.biggest_label.set_hand_thread(hand_thread)

    def connect_hand_thread(self):
        # here you pass just the first mouse move of the first page when we need to pass the mouseMove the page we are in since we always do another cursor when new page
        self.hand_thread.coor_signal.connect(self.biggest_label.mouseMove)
        self.hand_thread.click_signal.connect(self.biggest_label.mouseClick)
        
    def increase_font(self, e):
        self.drawing_mask.font_size += 3 
        self.font_size_label.setText(f"font size : {self.drawing_mask.font_size}")
    def decrease_font(self, e):
        if self.drawing_mask.font_size > 2:
            self.drawing_mask.font_size -= 3 
        self.font_size_label.setText(f"font size : {self.drawing_mask.font_size}")
        
    
    def color_layout(self):
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignCenter)
        hboxes = [QHBoxLayout(),QHBoxLayout(),QHBoxLayout(),QHBoxLayout(),QHBoxLayout(),QHBoxLayout(),QHBoxLayout()]
        colors = ['red','darkRed','green', 'darkGreen','yellow','orange','blue','darkBlue','black','white','cyan','darkCyan','magenta','darkMagenta']
        funcs = [self.color_red,self.color_darkRed,self.color_green,self.color_darkGreen,
                 self.color_yellow,self.color_orange,
                 self.color_blue,self.color_darkBlue,self.color_black,self.color_white,self.color_cyan,
                 self.color_darkCyan,self.color_magenta,self.color_darkMagenta]
        color = 0
        for hbox in hboxes:
            vbox.addLayout(hbox)
            button = QPushButton(colors[color])
            button.setStyleSheet(f"background-color: {colors[color]}; color: {colors[color]};")
            button.clicked.connect(funcs[color])
            hbox.addWidget(button)
            self.widgets.append(button)
            color += 1
            button = QPushButton(colors[color])
            button.setStyleSheet(f"background-color: {colors[color]}; color: {colors[color]};")
            button.clicked.connect(funcs[color])
            hbox.addWidget(button)
            self.widgets.append(button)
            color += 1
        
                    
        button = QPushButton("Eraser")
        button.clicked.connect(self.erase)
        self.widgets.append(button)
        vbox.addWidget(button)
            
        return vbox
    
    def erase(self):
        self.drawing_mask.font_color = "transparent"
    def color_red(self):
        self.drawing_mask.font_color = "red"
    def color_darkRed(self):
        self.drawing_mask.font_color = "darkRed"
    def color_blue(self):
        self.drawing_mask.font_color = "blue"
    def color_darkBlue(self):
        self.drawing_mask.font_color = "darkBlue"
    def color_green(self):
        self.drawing_mask.font_color = "green"
    def color_darkGreen(self):
        self.drawing_mask.font_color = "darkGreen"
    def color_white(self):
        self.drawing_mask.font_color = "white"
    def color_black(self):
        self.drawing_mask.font_color = "black"
    def color_cyan(self):
        self.drawing_mask.font_color = "cyan"
    def color_darkCyan(self):
        self.drawing_mask.font_color = "darkCyan"
    def color_magenta(self):
        self.drawing_mask.font_color = "magenta"
    def color_darkMagenta(self):
        self.drawing_mask.font_color = "darkMagenta"
    def color_yellow(self):
        self.drawing_mask.font_color = "yellow"
    def color_orange(self):
        self.drawing_mask.font_color = "orange"

    
        
    def save_page(self):
        if self.pno < self.pages.__len__():
            self.pages[self.pno].mask = self.drawing_mask
        else:
            self.pages.append(Page(self.drawing_mask,self.pno))
        
    def next(self,e):
        self.save_page()
        self.stack_drawing.removeWidget(self.label)
        self.stack_drawing.removeWidget(self.drawing_mask)
        self.pno += 1

            
        self.pixmap = self.get_pix_page(self.doc)
        self.label = self.making_canvas()
        if self.pno < self.pages.__len__():
            self.drawing_mask = self.pages[self.pno].mask
        else:
            self.drawing_mask = DrawingMask(self.label)
        self.biggest_label.changeVir(self.drawing_mask)
        self.stack_drawing.addWidget(self.drawing_mask)
        self.stack_drawing.addWidget(self.label)
        
        
    def previous(self,e):
        self.save_page()
        self.stack_drawing.removeWidget(self.label)
        self.stack_drawing.removeWidget(self.drawing_mask)
        self.pno -= 1

            
        self.pixmap = self.get_pix_page(self.doc)
        self.label = self.making_canvas()
        if self.pno < self.pages.__len__():
            self.drawing_mask = self.pages[self.pno].mask
        else:
            self.drawing_mask = DrawingMask(self.label)
        self.biggest_label.changeVir(self.drawing_mask)
        self.stack_drawing.addWidget(self.drawing_mask)
        self.stack_drawing.addWidget(self.label)
        
        
    def blank_page(self):
        self.save_page()
        self.stack_drawing.removeWidget(self.label)
        self.stack_drawing.removeWidget(self.drawing_mask)
        self.pno += 1      
        canvas = QPixmap(self.label.width(), self.label.height())
        canvas.fill(Qt.white)
        self.label = VirtualCursor(canvas)
        self.drawing_mask = DrawingMask(self.label)
        self.biggest_label.changeVir(self.drawing_mask)
        self.stack_drawing.addWidget(self.drawing_mask)
        self.stack_drawing.addWidget(self.label)
        
        
    def get_pix_page(self,doc):
        page = doc[self.pno]
        zoom = 2   # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pixmap_image = page.get_pixmap(matrix = mat)
        bytes = QImage.fromData(pixmap_image.tobytes())
        pixmap = QPixmap.fromImage(bytes) 
        return pixmap
    
    def making_canvas(self):
        label = VirtualCursor(self.pixmap)
        self.setMinimumSize(label.size())
        return label
        


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
        self.setWindowTitle("hand_window")
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
            cv_img, self.image.size().width(), self.image.size().height()
        )
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
        print("1")
        super().__init__()
        print("2")
        # setting geometry
        self.setGeometry(100, 100, 800, 600)

        # setting style sheet
        self.setStyleSheet("background : lightgrey;")
        print("3")
        # getting available cameras
        self.available_cameras = QCameraInfo.availableCameras()
        print(self.available_cameras)
        print("4")
        # if no camera found
        if not self.available_cameras:
            # exit the code
            print("no camera detected, exiting...")
            sys.exit()
        print("5")
        # creating a QCameraViewfinder object
        self.viewfinder = QCameraViewfinder()
        print("6")
        # creating a combo box for selecting camera
        self.camera_selector = QComboBox()
        print("7")
        # adding status tip to it
        self.camera_selector.setStatusTip("Choose desired camera")

        # adding tool tip to it
        self.camera_selector.setToolTip("Select Camera")
        self.camera_selector.setToolTipDuration(2500)

        # adding items to the combo box
        self.camera_selector.addItems(
            [camera.description() for camera in self.available_cameras]
        )

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
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))

        # start the camera
        self.camera.start()

        # creating a QCameraImageCapture object
        self.capture = QCameraImageCapture(self.camera)

        # showing alert if error occur
        self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg))

        # when image captured showing message
        self.capture.imageCaptured.connect(
            lambda d, i: self.status.showMessage(
                "Image captured : " + str(self.save_seq)
            )
        )

        # getting current camera name
        self.current_camera_name = self.available_cameras[i].description()

        # initial save sequence
        self.save_seq = 0

    # method for alerts
    def alert(self, msg):
        # error message
        error = QErrorMessage(self)

        # setting text to the error message
        error.showMessage(msg)

    def closeEvent(self, event):
        del self.camera
        CFG.camIdx = self.camera_selector.currentIndex()
        event.accept()


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
        self.left_screen.setFixedSize(self.width() // 2, self.height())
        self.right_screen.setFixedSize(self.width() // 2, self.height())

    def on_left_screen_clicked(self, event):
        CFG.mainScreen = 0
        self.hide()

    def on_right_screen_clicked(self, event):
        CFG.mainScreen = 1
        self.hide()
