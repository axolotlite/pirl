import sys
import io
import subprocess
import os

from handrec import Hand
# from pynput.keyboard import Key, Listener
from PyQt5.QtCore import Qt, QThread,QRect, pyqtSignal, QBuffer, QPoint
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
QPushButton, QHBoxLayout, QWidget, QDesktopWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QFont, QPainter, QColor, QPen
from fitz import *
#This needs to be initialized before pyqt to ensure removal of conflicting qt5 env vars

hand = Hand()

class VirtualCursor(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pressed = False
        self.x = 20
        self.y = 20      
        self.has_been_draw = False
        self.setMouseTracking(True)
        self.setCursor(Qt.BlankCursor)
        self._position = QPoint(self.x,self.y)
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

        painter.setBrush(QColor(255,255,0,int(self._opacity * 255)))
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
        

    def mouseMoveEvent(self, e):

        self._position = e.pos()
        if self.pressed == True:
            self.has_been_draw = True
            if self.last_x is None: # First event.
                self.last_x = e.x()
                self.last_y = e.y()

            painter = QPainter(self.pixmap())
            pen = QPen()
            pen.setWidth(5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
            painter.end()
            self.update()
            
            self.last_x = e.x()
            self.last_y = e.y()
        self.update()
    
    def mousePressEvent(self, e):
        self.pressed = True
        # print("pressed")


        
    def mouseReleaseEvent(self, e):
        # print("released")
        self.pressed = False
        self.last_x = None 
        self.last_y = None 

class pdf_window(QMainWindow):
    def __init__(self,document):
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
        self.layout.setContentsMargins(0,0,0,0)
        
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
        self.canvas = QPixmap(self.label.width(),self.label.height())
        self.canvas.fill(Qt.white) 
        self.label = self.making_canvas(self.canvas)
        self.layout.takeAt(0)
        self.layout.insertWidget(0, self.label)
        
    def get_pix_page(self,doc):
        page = doc[self.pno]
        zoom = 2    # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pixmap_image = page.get_pixmap(matrix = mat)
        bytes = QImage.fromData(pixmap_image.tobytes())
        pixmap = QPixmap.fromImage(bytes)
        
        return pixmap
    
    def making_canvas(self,pixmap):
        label = VirtualCursor()
        self.setMinimumSize(1000,900)
        self.pixmap = pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio)
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
        page = self.edited_pdf.new_page(self.pno - 1, self.doc[0].rect.width, self.doc[0].rect.height)
        page.show_pdf_page(self.doc[0].rect, copy, 0)
        pages = list (p for p in range(self.edited_pdf.page_count) if p != self.pno)
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
        page = self.edited_pdf.new_page(self.pno + 1, self.doc[0].rect.width, self.doc[0].rect.height)
        page.show_pdf_page(self.doc[0].rect, copy, 0)
        pages = list (p for p in range(self.edited_pdf.page_count) if p != self.pno + 2)
        self.edited_pdf.select(pages)
        self.edited_pdf.save("see.pdf")
        
    def move(self,str):
        print("move")
        self.label.mouseMoveEvent(str)
        self.update()
        self.has_been_draw = True
        
    def paintEvent(self, e):
        if self.label.pressed == True:
            self.has_been_draw = True
        
        
        
    def next(self):
        print(self.pno)
        if self.pno == self.doc.__len__() - 1:
            self.pno = 0
        if self.label.has_been_draw == True:
            self.pno +=1
            self.temp_pix = self.label.pixmap()
            pix = self.get_pix_page(self.doc)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)
            
            self.thread = MyThread(self,self.write_next) 
            self.thread.started.connect(lambda: print("writing started!"))
            self.thread.finished.connect(lambda: print("writing finished!"))
            self.thread.start()

            self.has_been_draw == False
        else:
            self.pno +=1
            pix = self.get_pix_page(self.edited_pdf)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)
        
    def previous(self):
        print(self.pno)
        if self.pno == 0:
            self.pno = self.doc.__len__() - 1
        if self.label.has_been_draw == True:
            self.temp_pix = self.label.pixmap()
            self.pno -=1
            pix = self.get_pix_page(self.edited_pdf)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)
            
            self.thread = MyThread(self,self.write_previous) 
            self.thread.started.connect(lambda: print("writing started!"))
            self.thread.finished.connect(lambda: print("writing finished!"))
            self.thread.start()

            self.has_been_draw == False
        else:
            self.pno -=1
            pix = self.get_pix_page(self.edited_pdf)
            self.label = self.making_canvas(pix)
            self.layout.takeAt(0)
            self.layout.insertWidget(0, self.label)
        
class MyThread(QThread):
    started = pyqtSignal()  # Custom signal for starting the thread
    finished = pyqtSignal()  # Custom signal for finishing the thread

    def __init__(self, pdf_window,func,parent=None):
        super().__init__(parent)
        self.func = func
        self.window = pdf_window         
    
    def run(self):
        self.started.emit()
        self.func()
        self.finished.emit()
        print(f"end of {self.func}")
        
class MainWindow(QMainWindow):
    def __init__(self,):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)


        self.button = QPushButton("choose a file")
        font = self.button.font()
        font.setPointSize(10)
        self.button.setFont(font)
        self.setFixedSize(800,500)

        self.calibration_button = QPushButton("Calibrate")
        self.calibration_button.setFont(font)
        self.calibration_button.clicked.connect(self.calibrate_screen)

        self.homography_button = QPushButton("manual homography")
        self.homography_button.setFont(font)
        self.homography_button.clicked.connect(self.manual_homography)

        label = QLabel("welcome to PIRL")
        label.setFont(QFont("Arial",20))
        layout.addWidget(label)
        layout.addWidget(self.button)
        layout.addWidget(self.calibration_button)
        layout.addWidget(self.homography_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.button.clicked.connect(self.choose_file)
        self.setCentralWidget(widget)
        
    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "PDF Files (*.pdf)", options=options)
        if file_name:
            print(f"Selected file: {file_name}")
            doc = fitz.open(file_name) 
            self.w = pdf_window(doc)
            screen = QDesktopWidget().screenGeometry(0)
            self.w.setGeometry(QRect(screen))
            self.show_new_window()
    def calibrate_screen(self):
        hand.main_loop(False)
    def manual_homography(self):
        hand.main_loop(True)
            
            
    # def keyboard_pressing(self):
    #     with Listener( on_press=self.on_press, on_release= None) as listener:
    #         listener.join()
    #     self.w.close()
            
    def show_new_window(self):
        self.w.showMaximized()
        
    def closeEvent(self, event):
        self.w.close()
        
# print(autocalibrate())
def main():
    # print(qt5_vars)
    # autocalibrator = Autocalibration()
    # autocalibrator.autocalibrate()
    #you have to delete qt5 variables before using qt5, i don't understand why.
    # Autocalibrator.delete_qt_vars()
    app = QApplication([])
    print("app")
    window = MainWindow()
    window.show()
    app.exec_()

    
if __name__ == '__main__':
    main()


