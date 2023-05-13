from PyQt5 import QtWidgets, QtCore
"""
inherits the Qlabel object to allow the addition of click, which triggers a callback function.
"""
class Label(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent=parent)
        self.mouse_click_callback = None
        self.setScaledContents(True)
    
    def setMouseCallback(self,func):
        self.mouse_click_callback = func

    def mousePressEvent(self, event):
        self.mouse_click_callback()
        self.clicked.emit()