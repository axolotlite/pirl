from PyQt5 import QtCore, QtGui, QtWidgets
from pyqt.qrcode_screen import QRScreen
from datetime import date
from modules.api.db_handler import DBHandler
def setSizePolicy(wPolicy,hPolicy):
    def policyFactory(policy):
        match policy.lower():
            case "minimum":
                return QtWidgets.QSizePolicy.Minimum
            case "fixed":
                return QtWidgets.QSizePolicy.Fixed
            case "exapanding":
                return QtWidgets.QSizePolicy.Expanding
            case "ignored":
                return QtWidgets.QSizePolicy.Ignored
    sizePolicy = QtWidgets.QSizePolicy(policyFactory(wPolicy), policyFactory(hPolicy))
    return sizePolicy

class StudentHeader():
    def __init__(self):
        self.button_array = []
        self.data_array = []
        self.className = None
        self.count = 0
        self.callback = None
        self.parameters = None
        self.qr_screen = QRScreen()
        self.db_handler = DBHandler()
        #we want to return this
        self.student_info_header_layout = QtWidgets.QHBoxLayout()
        #this is responsible for both button and label
        student_data_header_layout = QtWidgets.QHBoxLayout()
        student_name_placeholder_button = QtWidgets.QPushButton()
        student_name_placeholder_button.setEnabled(False)
        sizePolicy = setSizePolicy("ignored","fixed")
        print(sizePolicy)
        student_name_placeholder_button.setSizePolicy(sizePolicy)
        student_data_header_layout.addWidget(student_name_placeholder_button)
        student_id_placeholder_label = QtWidgets.QLabel()
        sizePolicy = setSizePolicy("ignored","fixed")
        student_id_placeholder_label.setSizePolicy(sizePolicy)
        student_id_placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        student_data_header_layout.addWidget(student_id_placeholder_label)
        student_data_header_layout.setStretch(0, 4)
        student_data_header_layout.setStretch(1, 2)
        self.student_info_header_layout.addLayout(student_data_header_layout)
        day_parent = QtWidgets.QHBoxLayout()
        day_parent.setContentsMargins(5, -1, -1, -1)
        self.day_layout = QtWidgets.QHBoxLayout()
        day_parent.addLayout(self.day_layout)
        day_parent.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        self.student_info_header_layout.addLayout(day_parent)
        self.student_info_header_layout.setStretch(0, 4)
        self.student_info_header_layout.setStretch(1, 5)
        student_name_placeholder_button.setText("StudentName")
        student_id_placeholder_label.setText("StudentId")

    def setCallback(self,parameters,callback):
        self.callback = callback
        self.parameters = parameters
    def addDay(self):
        button_count = len(self.button_array)
        # self.qr_screen.show()
        print(button_count)
        if( button_count > 0):
            lastIndex = len(self.button_array)
            prev_button = self.button_array[lastIndex - 1]
            prev_button.setText(str(lastIndex))
            # today = date.today().strftime('%d/%m/%Y')
            # add a new function on click
            prev_button.clicked.disconnect()
            prev_button.clicked.connect(lambda: self.qr_screen.setData(
                    date = self.data_array[lastIndex - 1]["date"],
                    className = self.className,
                    lessonId = self.data_array[lastIndex - 1]["lesson_number"]
                )
            )
        new_button = QtWidgets.QPushButton()
        self.button_array.append(new_button)
        sizePolicy = setSizePolicy("minimum","fixed")
        new_button.setSizePolicy(sizePolicy)
        new_button.setMinimumSize(QtCore.QSize(14, 15))
        new_button.setMaximumSize(QtCore.QSize(14, 25))
        new_button.setText("+")
        new_button.clicked.connect(lambda: self.setDays())
        self.day_layout.addWidget(new_button)
    def setDays(self, data=None, count=None):
        if(data != None):
            self.data_array = data
        if(count == None):
            print("hello there")
            self.callback()
            count = self.count
            count+=1
        self.count = count
        button_count = len(self.button_array)
        # print("comparing between: ", count, "and", button_count)
        #if the need number of buttons exceeds the present ones, add more buttons
        if(count >= button_count):
            print('count > button array, adding days')
            for _ in range(count - button_count + 1):
                self.addDay()
            return
        #if the needed number is less than needed, hide the newer ones
        else:
            print("the else statement")
            for index in range(button_count - 1):
                self.button_array[index].hide()
            for index in range(count):
                self.button_array[index].show()
    def getHeader(self):
        return self.student_info_header_layout

class StudentRow():
    def __init__(self,studentName, studentId, classId, attendance_records):
        self.studentName = studentName
        self.studentId = studentId
        self.classId = classId
        self.attendance_records = attendance_records
        self.db_handler = DBHandler()
        self.student_info_layout = QtWidgets.QHBoxLayout()
        self.student_data_layout = QtWidgets.QHBoxLayout()
        
        self.checkbox_parent = QtWidgets.QHBoxLayout()
        self.checkbox_parent.setContentsMargins(5, -1, -1, -1)

        self.checkbox_layout = QtWidgets.QHBoxLayout()

        self.checkboxInit = False
        self.checkboxCount = 0
        self.addStudentInfo(studentName,studentId)
        print("student row attendance recs", attendance_records)
        for record in attendance_records:
            self.addCheckBox(record)
        
        self.checkbox_parent.addLayout(self.checkbox_layout)
        self.checkbox_parent.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        
        self.student_info_layout.addLayout(self.student_data_layout)
        self.student_info_layout.addLayout(self.checkbox_parent)
        self.student_info_layout.setStretch(0, 4)
        self.student_info_layout.setStretch(1, 5)

    def addCheckBox(self, checked):
        attendance_checkBox = QtWidgets.QCheckBox()
        sizePolicy = setSizePolicy("fixed","fixed")
        attendance_checkBox.setSizePolicy(sizePolicy)
        attendance_checkBox.setText("")
        attendance_checkBox.setChecked(checked)
        self.checkbox_layout.addWidget(attendance_checkBox)
        self.checkboxCount+=1
        idx = self.checkboxCount
        print(idx)
        attendance_checkBox.stateChanged.connect(
            lambda: self.db_handler.set_attendance(
                self.studentId,
                self.classId,
                idx, 
                attendance_checkBox.isChecked()
            )
        )

    def addStudentInfo(self, studentName, studentId):
        #button creation
        student_name_button = QtWidgets.QPushButton()
        student_name_button.setText(studentName)
        sizePolicy = setSizePolicy("ignored","fixed")
        student_name_button.setSizePolicy(sizePolicy)
        #label creation
        student_id_label = QtWidgets.QLabel()
        student_id_label.setText(studentId)
        sizePolicy = setSizePolicy("ignored","fixed")
        student_id_label.setSizePolicy(sizePolicy)
        student_id_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.student_data_layout.addWidget(student_name_button)
        self.student_data_layout.addWidget(student_id_label)

        self.student_data_layout.setStretch(0, 4)
        self.student_data_layout.setStretch(1, 2)
    
    def getRow(self):
        return self.student_info_layout
    def delete(self,qObject=None):
        # self.student_info_layout = None
        #recursively delete objects inside of this row
        #parent object
        if(qObject == None):
            qObject = self.student_info_layout
        #if the object is qSpacers
        elif(isinstance(qObject, QtWidgets.QSpacerItem)):
            print(qObject)
            return qObject
        #if the object is a layout
        while qObject.count():
            child = qObject.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            else:
                #if it's a spacer, it'll get returned, then deleted
                qObject.removeItem(self.delete(child))
        del self
if __name__ == "__main__":
    row = StudentRow("test","1234")