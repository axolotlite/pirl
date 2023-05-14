import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QDesktopWidget
from PyQt5.QtGui import QFont, QResizeEvent, QPainter
from PyQt5.QtCore import Qt, QRectF, QTimer, QSize
from Clickable_Label import Label

class NumberWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Get screen size
        screen_size = QApplication.primaryScreen().size()
        width = int(screen_size.width() / 10)
        height = int(screen_size.height() / 10)
        widget_size = QSize(width, height)

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        font = QFont("Arial", 100, QFont.Bold)

        self.label = Label(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(font)
        self.label.setStyleSheet("color: white;")

        # Set widget size and position
        self.resize(widget_size)

    def update_number(self, number):
        self.label.setText(str(number))
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

        size = min(self.width(), self.height())
        self.setGeometry(0, 0, size, size)
        self.label.setGeometry(self.rect())

    def paintEvent(self, event: QResizeEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        size = min(self.width(), self.height())
        rect = QRectF(0, 0, size, size)
        painter.drawEllipse(rect)
        painter.fillRect(rect, Qt.transparent)
        
    def center(self,screen_number):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().screenGeometry(screen_number).center()
        # Offset the window a bit higher
        centerPoint.setY(centerPoint.y())
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    widget = NumberWidget()
    widget.update_number(42)
    widget.show()

    sys.exit(app.exec_())