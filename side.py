import sys
import cv2
import os
from sys import platform
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))


try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
#import openpose


def main():
    try:
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../openpose/models/"

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item
        
        
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread("media/side.jpg")
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        
        #Change to tuple
        head = [datum.poseKeypoints[0][17][0],datum.poseKeypoints[0][17][1]]
        neck = [datum.poseKeypoints[0][1][0],datum.poseKeypoints[0][1][1]]
        body = [(datum.poseKeypoints[0][1][0] + datum.poseKeypoints[0][8][0])/2, (datum.poseKeypoints[0][1][1] +datum.poseKeypoints[0][8][1])/2]
        pelvis = [datum.poseKeypoints[0][9][0],datum.poseKeypoints[0][9][1]]
        shoulder = [datum.poseKeypoints[0][2][0],datum.poseKeypoints[0][2][1]]
        arm = [(datum.poseKeypoints[0][2][0] + datum.poseKeypoints[0][3][0])/2, (datum.poseKeypoints[0][2][1] +datum.poseKeypoints[0][3][1])/2]
        forearm = [(datum.poseKeypoints[0][3][0] + datum.poseKeypoints[0][4][0])/2, (datum.poseKeypoints[0][3][1] +datum.poseKeypoints[0][4][1])/2]
        hand = [datum.poseKeypoints[0][4][0],datum.poseKeypoints[0][4][1]]
        thigh = [(datum.poseKeypoints[0][9][0] + datum.poseKeypoints[0][10][0])/2, (datum.poseKeypoints[0][9][1] +datum.poseKeypoints[0][10][1])/2]
        shank = [(datum.poseKeypoints[0][10][0] + datum.poseKeypoints[0][11][0])/2, (datum.poseKeypoints[0][10][1] +datum.poseKeypoints[0][11][1])/2]
        foot = [(datum.poseKeypoints[0][22][0] + datum.poseKeypoints[0][24][0])/2, (datum.poseKeypoints[0][22][1] +datum.poseKeypoints[0][24][1])/2]
        
        print("head", head)
        print("neck", neck)
        print("body", body)
        print("pelvis", pelvis)
        print("shoulder", shoulder)
        print("arm", arm)
        print("forearm", forearm)
        print("hand", hand)
        print("thigh", thigh)
        print("shank", shank)
        print("foot", foot)

        COM_x = \
            head[0]*0.05 + \
            neck[0]*0.03 + \
            body[0]*0.27 + \
            pelvis[0]*0.11 + \
            shoulder[0]*0.05*2 + \
            arm[0]*0.03*2 + \
            forearm[0]*0.02*2 + \
            hand[0]*0.01*2 + \
            thigh[0]*0.1*2 + \
            shank[0]*0.04*2 + \
            foot[0]*0.02*2
        
        COM_y = \
            head[1]*0.05 + \
            neck[1]*0.03 + \
            body[1]*0.27 + \
            pelvis[1]*0.11 + \
            shoulder[1]*0.05*2 + \
            arm[1]*0.03*2 + \
            forearm[1]*0.02*2 + \
            hand[1]*0.01*2 + \
            thigh[1]*0.1*2 + \
            shank[1]*0.04*2 + \
            foot[1]*0.02*2
        print(COM_x)
        print(COM_y)

        COM = (int(COM_x), int(COM_y))
        radius = 20
        # Blue color in BGR 
        color = (255, 0, 0) 
        thickness = 2
        #Add COM circle to image
        cv2.circle(imageToProcess, COM, radius, color, thickness)
        # Find better way to readd data to datum
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)


    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == '__main__':
    main()
