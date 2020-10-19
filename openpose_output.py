import sys
import cv2
import os
import argparse
import numpy as np
import math
import copy
import scipy.signal
from patient import Patient
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QProgressBar, QPushButton, QApplication
from PyQt5 import QtCore
from tqdm import tqdm
import time
import json
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))

try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable \
        `BUILD_PYTHON` in CMake and have this Python script in the right \
        folder?')
    raise e

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
        while count < TIME_LIMIT:
            time.sleep(0.1)
            count +=1
            self.countChanged.emit(self.frame_counter.progress)
            if self.frame_counter.progress == 100:
                break

def get_keypoints(inputvid, model, progressUI):
    try:
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = model
        #Find a better way to do this. Currently saves each frame as json
        #params["write_json"] = "json_output"

        #Configure Openpose Python Wrapper
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        vid_location = inputvid
        cap = cv2.VideoCapture(vid_location)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_num = 0
        pbar = tqdm(total=100)
        progress = (1 / total_frames) * 100
        fc = Frame_Counter(total_frames)
        progressUI.start(External(fc))


        #lists to store data per frame 
        json_data = {}
        json_data['openpose'] = []
        date = datetime.datetime.now()
        file_name = 'json_output/'+time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))+ '.json'
        
        print("Generating Pose")
        while(cap.isOpened()):
            ret, frame = cap.read()
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if ret == True:
                # Process Image
                datum = op.Datum()
                imageToProcess = frame

                frame_num += 1
                fc.incre_frame()
                QApplication.processEvents() 

                pbar.update(progress)
                
                cv2.putText(imageToProcess, str(frame_num), (100,100), font, 1, 
                    (255,255,255), 1)
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])
                
                json_data['openpose'].append({frame_num:{
                    'Nose': [float(datum.poseKeypoints[0][0][0]),float(datum.poseKeypoints[0][0][1])],
                    'Neck': [float(datum.poseKeypoints[0][1][0]),float(datum.poseKeypoints[0][1][1])],
                    'RShoulder': [float(datum.poseKeypoints[0][2][0]),float(datum.poseKeypoints[0][2][1])],
                    'RElbow': [float(datum.poseKeypoints[0][3][0]),float(datum.poseKeypoints[0][3][1])],
                    'RWrist': [float(datum.poseKeypoints[0][4][0]),float(datum.poseKeypoints[0][4][1])],
                    'LShoulder': [float(datum.poseKeypoints[0][5][0]),float(datum.poseKeypoints[0][5][1])],
                    'LElbow': [float(datum.poseKeypoints[0][6][0]),float(datum.poseKeypoints[0][6][1])],
                    'LWrist': [float(datum.poseKeypoints[0][7][0]),float(datum.poseKeypoints[0][7][1])],
                    'MidHip': [float(datum.poseKeypoints[0][8][0]),float(datum.poseKeypoints[0][8][1])],
                    'RHip': [float(datum.poseKeypoints[0][9][0]),float(datum.poseKeypoints[0][9][1])],
                    'RKnee': [float(datum.poseKeypoints[0][10][0]),float(datum.poseKeypoints[0][10][1])],
                    'RAnkle': [float(datum.poseKeypoints[0][11][0]),float(datum.poseKeypoints[0][11][1])],
                    'LHip': [float(datum.poseKeypoints[0][12][0]),float(datum.poseKeypoints[0][12][1])],
                    'LKnee': [float(datum.poseKeypoints[0][13][0]),float(datum.poseKeypoints[0][13][1])],
                    'LAnkle': [float(datum.poseKeypoints[0][14][0]),float(datum.poseKeypoints[0][14][1])],
                    'REye': [float(datum.poseKeypoints[0][15][0]),float(datum.poseKeypoints[0][15][1])],
                    'LEye': [float(datum.poseKeypoints[0][16][0]),float(datum.poseKeypoints[0][16][1])],
                    'REar': [float(datum.poseKeypoints[0][17][0]),float(datum.poseKeypoints[0][17][1])],
                    'LEar': [float(datum.poseKeypoints[0][18][0]),float(datum.poseKeypoints[0][18][1])],
                    'LBigToe': [float(datum.poseKeypoints[0][19][0]),float(datum.poseKeypoints[0][19][1])],
                    'LSmallToe': [float(datum.poseKeypoints[0][20][0]),float(datum.poseKeypoints[0][20][1])],
                    'LHeel': [float(datum.poseKeypoints[0][21][0]),float(datum.poseKeypoints[0][21][1])],
                    'RBigToe': [float(datum.poseKeypoints[0][22][0]),float(datum.poseKeypoints[0][22][1])],
                    'RSmallToe': [float(datum.poseKeypoints[0][23][0]),float(datum.poseKeypoints[0][23][1])],
                    'RHeel': [float(datum.poseKeypoints[0][24][0]),float(datum.poseKeypoints[0][24][1])]
                }})


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        #out.release()
        cv2.destroyAllWindows
        pbar.close
        print("Generating Json")
        with open(file_name, 'w') as outfile:
            json.dump(json_data, outfile, indent=4)
        return file_name
        
    except Exception as e:
        print(e)
        sys.exit(-1)

# if __name__ == '__main__':
#     main()
