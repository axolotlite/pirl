# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/menu_template.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from pyqt.student_row import StudentRow, StudentHeader, setSizePolicy
from pyqt.student_addition_screen import AdditionScreen
import sys,os
sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
from modules.api.db_handler import DBHandler
from modules.api.pirl_api import APIWrapper
from datetime import date
class ClassroomCompanion(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.subject_buttons = []
        self.classes = []
        self.students = []
        self.db_handler = DBHandler()
        self.add_student_screen = AdditionScreen()
        self.addition_layout = None
        self.student_header = StudentHeader()
        self.api = APIWrapper()
        self.api.start()
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Classroom Companion")
        self.resize(829, 564)
        # self.add_student_screen.set_callback(self.addNewStudent)
        #main widget
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.subject_menu_layout = QtWidgets.QVBoxLayout()
        self.menu_label = QtWidgets.QLabel()
        sizePolicy = setSizePolicy("minimum","fixed")
        self.menu_label.setSizePolicy(sizePolicy)
        self.menu_label.setAlignment(QtCore.Qt.AlignCenter)
        self.subject_menu_layout.addWidget(self.menu_label)
        self.subject_buttons_layout = QtWidgets.QVBoxLayout()
        sizePolicy = setSizePolicy("ignored","fixed")
        self.subject_menu_layout.addLayout(self.subject_buttons_layout)
        self.add_subject_button = QtWidgets.QPushButton()
        self.add_subject_button.clicked.connect(self.addNewSubject)
        sizePolicy = setSizePolicy("ignored","fixed")
        self.add_subject_button.setSizePolicy(sizePolicy)
        self.subject_menu_layout.addWidget(self.add_subject_button)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.subject_menu_layout.addItem(spacerItem)
        self.subject_menu_layout.setStretch(0, 1)
        self.horizontalLayout.addLayout(self.subject_menu_layout)
        self.subject_info_layout = QtWidgets.QVBoxLayout()
        # self.student_header = StudentHeader()
        self.student_header.setCallback((),self.addDay)
        self.student_info_header_layout = self.student_header.getHeader()
        self.subject_info_layout.addLayout(self.student_info_header_layout)

        self.students_info = QtWidgets.QVBoxLayout()
        self.subject_info_layout.addLayout(self.students_info)
        self.add_student_button = QtWidgets.QPushButton()
        self.add_student_button.clicked.connect(lambda: self.addNewStudent())
        self.add_student_button.setEnabled(False)
        self.subject_info_layout.addWidget(self.add_student_button)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.subject_info_layout.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.subject_info_layout)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        
        self.setLayout(self.horizontalLayout)
        # self.clearSubjects()
        self.menu_label.setText("Class rooms")
        self.add_subject_button.setText("+")
        self.add_student_button.setText("+")
        # self.retranslateUi()
        # QtCore.QMetaObject.connectSlotsByName()
        #initialize UI
        self.loadSubjects()


    def clearSubjects(self):
        #the button cannot delete itself.
        limit = self.subject_buttons_layout.count()
        for item in self.subject_buttons:
            # item = self.subject_buttons_layout.itemAt(i)
            print(item)
            self.subject_buttons_layout.removeWidget(item)
        self.subject_buttons = []
        self.subject_buttons_layout.update()
    def clearStudents(self):
        print("pre loop: ",self.students)
        for student in self.students: 
            student.delete()
        self.students = []
        self.students_info.update()
        self.subject_info_layout.update()
    def addStudent(self, studentName, studentId, classId, attendanceRecords):
        row = StudentRow(studentName,studentId, classId, attendanceRecords)
        self.students.append(row)
        self.students_info.addLayout(row.getRow())
    def addNewStudent(self):
        #==== new implementation
        def button_action(student_name_textbox, student_id_textbox):
            student_name = student_name_textbox.text()
            student_id = student_id_textbox.text()
            print("these stuff here: ",student_name,student_id)
            class_id = self.classId
            class_name = self.className
            if(student_name.replace(" ", "") == "" or student_id.replace(" ", "") == ""):
                return
            record_number = self.student_header.count
            print(record_number)
            self.db_handler.add_student(student_id,student_name)
            for lesson_number in range(1,record_number + 1):
                self.db_handler.add_attendance(student_id, class_id, lesson_number, False)
            student_name_textbox.deleteLater()
            student_id_textbox.deleteLater()
            # self.clearStudents()
            self.addStudents(class_name,class_id)
        
        if(self.addition_layout != None and isinstance(self.addition_layout,QtWidgets.QHBoxLayout)):
            student_name_textbox = self.addition_layout.itemAt(0).widget()
            student_id_textbox = self.addition_layout.itemAt(1).widget()
            print(type(student_name_textbox))
            print(type(student_id_textbox))
            button_action(student_name_textbox,student_id_textbox)
            return
        student_name_textbox = QtWidgets.QLineEdit()
        student_id_textbox = QtWidgets.QLineEdit()
        self.addition_layout = QtWidgets.QHBoxLayout()
        self.addition_layout.addWidget(student_name_textbox)
        self.addition_layout.addWidget(student_id_textbox)
        self.students_info.addLayout(self.addition_layout)
        student_name_textbox.returnPressed.connect(lambda: button_action(student_name_textbox,student_id_textbox))
        student_id_textbox.returnPressed.connect(lambda: button_action(student_name_textbox,student_id_textbox))
    def addStudents(self, className, classId):
        self.add_student_button.setEnabled(True)
        self.clearStudents()
        self.classId = classId
        self.className = className
        self.add_student_screen.class_id = classId
        students = self.db_handler.get_students(classId)
        print(classId)
        max_record = 0
        for student in students:
            print(student)
            studentName, studentId = student.get('student_name'),str(student.get('student_id'))
            attendanceRecords = student.get('attendance_records')
            self.addStudent(studentName, studentId, classId, attendanceRecords)
        
        recorded_days = self.db_handler.get_lessons(classId)
        self.student_header.setDays(count=len(recorded_days), data=recorded_days)
        self.student_header.className = className
        self.student_header.classId = classId
    def addDay(self):
        today = date.today().strftime('%Y-%m-%d')
        self.db_handler.add_lesson(today,self.classId)
        for row in self.students:
            print('added stuff')
            self.db_handler.add_attendance(row.studentId, self.classId, -1, False)
            row.addCheckBox(False)
            print(row)
    def addSubject(self,subjectName, subjectId):
        subject_button = QtWidgets.QPushButton()
        subject_button.setText(subjectName)
        sizePolicy = setSizePolicy("ignored","fixed")
        subject_button.setSizePolicy(sizePolicy)
        subject_button.clicked.connect(lambda: self.addStudents(subjectName,subjectId))
        self.subject_buttons.append(subject_button)
        self.subject_buttons_layout.addWidget(subject_button)
    def addNewSubject(self):
        def button_action(subject_name_textbox):
            subject_name = subject_name_textbox.text()
            if(subject_name.replace(" ", "") == ""):
                return
            self.db_handler.add_class(subject_name,subject_name)
            subject_name_textbox.deleteLater()
            self.clearSubjects()
            self.loadSubjects()
        if(self.subject_buttons != [] and isinstance(self.subject_buttons[-1],QtWidgets.QLineEdit)):
            button_action(self.subject_buttons[-1])
            return
        subject_name_textbox = QtWidgets.QLineEdit()
        self.subject_buttons.append(subject_name_textbox)
        self.subject_buttons_layout.addWidget(subject_name_textbox)
        subject_name_textbox.returnPressed.connect(lambda: button_action(subject_name_textbox))
    # def retranslateUi(self, Form):
    #     _translate = QtCore.QCoreApplication.translate
    #     Form.setWindowTitle(_translate("Form", "Form"))
        
    def loadSubjects(self):
        classes = self.db_handler.get_classes()
        self.classes = classes
        # print(classes)
        for item in classes:
            subjectName = item.get('subject')
            subjectId = item.get('id')
            self.addSubject(subjectName, subjectId)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
