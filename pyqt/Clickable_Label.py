from PyQt5 import QtWidgets, QtCore

"""
inherits the Qlabel object to allow the addition of click, which triggers a callback function.
"""


class Label(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None, func=None, params=None):
        QtWidgets.QLabel.__init__(self, parent=parent)
        self.mouse_click_callback = func
        self.mouse_click_callback_params = params
        self.setScaledContents(True)

    def setMouseCallback(self, func):
        self.mouse_click_callback = func

    def setCallbackParams(self, params):
        self.mouse_click_callback_params = params

    def mousePressEvent(self, event):
        if self.mouse_click_callback_params == None:
            self.mouse_click_callback()
        else:
            self.mouse_click_callback(self.mouse_click_callback_params)
        self.clicked.emit()
