import vid_pose 
import openpose_output
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import progress

class progressView(QtWidgets.QWidget):
    def __init__(self):
        super(progressView, self).__init__()
        #self.theText = QtWidgets.QLabel('test', self )
        self.progressVal = 0
        self.progressBar = QtWidgets.QProgressBar(self)
        self.progressBar.setGeometry(QtCore.QRect(20, 70, 230, 23))
        self.progressBar.setProperty("value", self.progressVal)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setRange(0, 100)
        self.step = 0
        self.setWindowTitle('Openpose Progress')

    def setProgress(self, val):
        self.progressBar.setValue(val)

    def start(self, thread):
        #self.calc = External()
        self.calc = thread
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.start()

    def onCountChanged(self, value):
        self.progressBar.setValue(value)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(559, 468)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)

        # self.progressVal = 0
        # self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        # self.progressBar.setGeometry(QtCore.QRect(20, 70, 230, 23))
        # self.progressBar.setProperty("value", self.progressVal)
        # self.progressBar.setObjectName("progressBar")
        # self.progressBar.setRange(0, 150)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 0, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(170, 40, 281, 20))
        self.lineEdit.setObjectName("lineEdit")

        self.inputVidButton = QtWidgets.QPushButton(self.centralwidget)
        self.inputVidButton.setGeometry(QtCore.QRect(460, 40, 75, 23))
        self.inputVidButton.setObjectName("inputVidButton")
        self.inputVidButton.clicked.connect(self.inputFileNameDialog)

        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(170, 80, 281, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.modelButton = QtWidgets.QPushButton(self.centralwidget)
        self.modelButton.setGeometry(QtCore.QRect(460, 80, 75, 23))
        self.modelButton.setObjectName("modelButton")
        self.modelButton.clicked.connect(self.modelFileNameDialog)

        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(58, 40, 111, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(23, 80, 145, 16))
        self.label_3.setObjectName("label_3")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(58, 120, 111, 16))
        self.label_13.setObjectName("label_13")
        self.frontButton = QtWidgets.QRadioButton(self.layoutWidget1)
        self.frontButton.setGeometry(QtCore.QRect(200, 120, 82, 17))
        self.frontButton.setObjectName("frontButton")
        self.sideButton = QtWidgets.QRadioButton(self.layoutWidget1)
        self.sideButton.setGeometry(QtCore.QRect(300, 120, 82, 17))
        self.sideButton.setObjectName("sideButton")


        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(200, 150, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 200, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.maleButton = QtWidgets.QRadioButton(self.layoutWidget2)
        self.maleButton.setGeometry(QtCore.QRect(50, 230, 82, 17))
        self.maleButton.setObjectName("maleButton")
        self.femaleButton = QtWidgets.QRadioButton(self.layoutWidget2)
        self.femaleButton.setGeometry(QtCore.QRect(50, 250, 82, 17))
        self.femaleButton.setObjectName("femaleButton")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(250, 200, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(430, 200, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(230, 320, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(47, 370, 121, 16))
        self.label_9.setObjectName("label_9")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(170, 370, 281, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.outputVidButton = QtWidgets.QPushButton(self.centralwidget)
        self.outputVidButton.setGeometry(QtCore.QRect(460, 370, 75, 23))
        self.outputVidButton.setObjectName("outputVidButton")
        self.outputVidButton.clicked.connect(self.outputFileNameDialog)
        #self.label_10 = QtWidgets.QLabel(self.centralwidget)
        #self.label_10.setGeometry(QtCore.QRect(27, 340, 140, 16))
        #self.label_10.setObjectName("label_10")
        #self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        #self.lineEdit_4.setGeometry(QtCore.QRect(170, 340, 281, 20))
        #self.lineEdit_4.setObjectName("lineEdit_4")
        self.generateButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateButton.setGeometry(QtCore.QRect(230, 420, 75, 23))
        self.generateButton.setObjectName("generateButton")
        self.generateButton.clicked.connect(self.generateAction)
        self.heightInput = QtWidgets.QLineEdit(self.centralwidget)
        self.heightInput.setGeometry(QtCore.QRect(240, 240, 71, 20))
        self.heightInput.setObjectName("heightInput")
        self.weightInput = QtWidgets.QLineEdit(self.centralwidget)
        self.weightInput.setGeometry(QtCore.QRect(410, 240, 71, 20))
        self.weightInput.setObjectName("weightInput")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(315, 245, 47, 13))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(485, 245, 47, 18))
        self.label_12.setObjectName("label_12")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 559, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Input"))
        self.inputVidButton.setText(_translate("MainWindow", "Browse"))
        self.modelButton.setText(_translate("MainWindow", "Broswse"))
        self.label_2.setText(_translate("MainWindow", "Input Video Dir:"))
        self.label_3.setText(_translate("MainWindow", "Openpose Model Dir:"))
        self.label_4.setText(_translate("MainWindow", "Patient Data"))
        self.label_5.setText(_translate("MainWindow", "Gender"))
        self.maleButton.setText(_translate("MainWindow", "Male"))
        self.femaleButton.setText(_translate("MainWindow", "Female"))
        self.label_6.setText(_translate("MainWindow", "Height"))
        self.label_7.setText(_translate("MainWindow", "Weight"))
        self.label_8.setText(_translate("MainWindow", "Output"))
        self.label_9.setText(_translate("MainWindow", "Output Video Dir:"))
        self.outputVidButton.setText(_translate("MainWindow", "Browse"))
        #self.label_10.setText(_translate("MainWindow", "Output Video Name:"))
        self.generateButton.setText(_translate("MainWindow", "Generate"))
        self.label_11.setText(_translate("MainWindow", "cm"))
        self.label_12.setText(_translate("MainWindow", "kg"))
        self.label_13.setText(_translate("MainWindow", "Orientation:"))
        self.frontButton.setText(_translate("MainWindow", "Front"))
        self.sideButton.setText(_translate("MainWindow", "Side"))
    
    def inputFileNameDialog(self):
        """Select Input Video button event

        Opens current directory and allows selection of avi and mp4 video files.
        """
        fileName = str(QtWidgets.QFileDialog.getOpenFileName(None, 
            "Select Video", os.getcwd(), "Video files (*.avi *.mp4)" ))
        if fileName:
            dir = fileName.split(",")
            start = dir[0].find("'") + len("'")
            directory = dir[0][start:len(dir[0])-1]
            self.lineEdit.setText(directory)

    def modelFileNameDialog(self):
        """Select Openpose model folder button event

        Opens home directory and used to select openpose models folder
        """
        fileName = str(QtWidgets.QFileDialog.getExistingDirectory(None, 
            "Select Directory" ,"/home"))
        if fileName:
            self.lineEdit_2.setText(fileName)
    
    def outputFileNameDialog(self):
        """Select output video folder button event

        Opens current directory and used to select where the output video should
        be saved
        """
        fileName = str(QtWidgets.QFileDialog.getExistingDirectory(None, 
            "Select Directory", os.getcwd()))
        if fileName:
            self.lineEdit_3.setText(fileName)
    
    def generateAction(self):
        """Generate Output

        Checks that all inputs are filled out and runs generate_output from 
        vid_pose.py
        """

        inputvid = self.lineEdit.text()
        model = self.lineEdit_2.text()

        orientation = None
        if self.frontButton.isChecked() == False:
            if self.sideButton.isChecked() == True:
                orientation = "side"
        else:
            orientation = "front"

        gender = None
        if self.maleButton.isChecked() == False:
            if self.femaleButton.isChecked() == False:
                print("No Gender Selected")
            else:
                gender = "female"
        else:
            gender = "male"
        outputvid = self.lineEdit_3.text()

        height = self.heightInput.text()
        weight = self.weightInput.text()
        
        if (inputvid == '' or model =='' or orientation is None or gender is None 
            or height == '' or weight =='' or outputvid == ''):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Not all information has been filled out')
            msg.setWindowTitle("Error")
            msg.exec_()
        else:
            height = int(self.heightInput.text())
            weight = int(self.weightInput.text())
            self.progressView = progressView()
            self.progressView.show()
            file_name = openpose_output.get_keypoints(inputvid, model, self.progressView)
            print(inputvid)
            print(model)
            print(orientation)
            print(gender)
            print(outputvid)
            print(file_name)
            vid_pose.generate_output(inputvid, model, orientation, gender, height, weight, outputvid, file_name)
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())