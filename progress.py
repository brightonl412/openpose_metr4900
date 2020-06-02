import sys
from PyQt5.QtWidgets import QWidget, QProgressBar, QPushButton, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets
import time


class Frame_Counter():
    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.current_frame = 0
        self.progress = 0

    def incre_frame(self):
        self.current_frame += 1
        self.progress = (self.current_frame/self.max_frames)*100

TIME_LIMIT = 2400000000000000000000000000
class External(QtCore.QThread):
    """
    Runs a counter thread.
    """
    countChanged = QtCore.pyqtSignal(int)
    def __init__(self, frame_counter):
        super(External, self).__init__()
        self.frame_counter = frame_counter

    def run(self):
        count = 0
        while count < 100:
            time.sleep(0.5)
            count +=1
            print(count)
            self.countChanged.emit(count)
            if self.frame_counter.progress == 100:
                break
