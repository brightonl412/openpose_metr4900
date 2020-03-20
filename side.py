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
gender = "male"

if gender == "male":
    body_perc = {
        "head": 0.0694,
        "body": 0.3229,
        "pelvis": 0.1117,
        "arm": 0.0271,
        "forearm": 0.0162,
        "hand": 0.0061,
        "thigh": 0.1416,
        "shank": 0.0433,
        "foot": 0.0137
    }
else:
    body_perc = {
        "head": 0.0668,
        "body": 0.301,
        "pelvis": 0.1247,
        "arm": 0.0255,
        "forearm": 0.0138,
        "hand": 0.0056,
        "thigh": 0.1478,
        "shank": 0.0481,
        "foot": 0.0129
    }




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
        imageToProcess = cv2.imread("media/side2.jpg")
        #Resize to show on screen maybe change to do at end, when displaying so that we have high acc 
        imageToProcess = cv2.resize(imageToProcess, (480,960))
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        
        #Change to tuple
        R_ear = [datum.poseKeypoints[0][17][0],datum.poseKeypoints[0][17][1]]
        body = [(datum.poseKeypoints[0][1][0] + datum.poseKeypoints[0][8][0])/2, (datum.poseKeypoints[0][1][1] +datum.poseKeypoints[0][8][1])/2]
        pelvis = [datum.poseKeypoints[0][8][0],datum.poseKeypoints[0][8][1]]
        R_arm = [(datum.poseKeypoints[0][2][0] + datum.poseKeypoints[0][3][0])/2, (datum.poseKeypoints[0][2][1] +datum.poseKeypoints[0][3][1])/2]
        R_forearm = [(datum.poseKeypoints[0][3][0] + datum.poseKeypoints[0][4][0])/2, (datum.poseKeypoints[0][3][1] +datum.poseKeypoints[0][4][1])/2]
        R_hand = [datum.poseKeypoints[0][4][0],datum.poseKeypoints[0][4][1]]
        R_thigh = [(datum.poseKeypoints[0][9][0] + datum.poseKeypoints[0][10][0])/2, (datum.poseKeypoints[0][9][1] +datum.poseKeypoints[0][10][1])/2]
        R_shank = [(datum.poseKeypoints[0][10][0] + datum.poseKeypoints[0][11][0])/2, (datum.poseKeypoints[0][10][1] +datum.poseKeypoints[0][11][1])/2]
        R_foot = [(datum.poseKeypoints[0][22][0] + datum.poseKeypoints[0][24][0])/2, (datum.poseKeypoints[0][22][1] +datum.poseKeypoints[0][24][1])/2]
        
        L_ear = [datum.poseKeypoints[0][18][0],datum.poseKeypoints[0][18][1]]
        L_arm = [(datum.poseKeypoints[0][5][0] + datum.poseKeypoints[0][6][0])/2, (datum.poseKeypoints[0][5][1] +datum.poseKeypoints[0][6][1])/2]
        L_forearm = [(datum.poseKeypoints[0][6][0] + datum.poseKeypoints[0][7][0])/2, (datum.poseKeypoints[0][6][1] +datum.poseKeypoints[0][7][1])/2]
        L_hand = [datum.poseKeypoints[0][7][0],datum.poseKeypoints[0][7][1]]
        L_thigh = [(datum.poseKeypoints[0][12][0] + datum.poseKeypoints[0][13][0])/2, (datum.poseKeypoints[0][12][1] +datum.poseKeypoints[0][13][1])/2]
        L_shank = [(datum.poseKeypoints[0][13][0] + datum.poseKeypoints[0][14][0])/2, (datum.poseKeypoints[0][13][1] +datum.poseKeypoints[0][14][1])/2]
        L_foot = [(datum.poseKeypoints[0][21][0] + datum.poseKeypoints[0][19][0])/2, (datum.poseKeypoints[0][21][1] +datum.poseKeypoints[0][19][1])/2]
        
        print("right ear", R_ear)
        print("right ear", L_ear)
        print("body", body)
        print("pelvis", pelvis)
        print("right arm", R_arm)
        print("left arm", L_arm)
        print("right forearm", R_forearm)
        print("left forearm", L_forearm)
        print("right hand", R_hand)
        print("left hand", L_hand)
        print("right thigh", R_thigh)
        print("left thigh", L_thigh)
        print("right shank", R_shank)
        print("left shank", L_shank)
        print("right foot", R_foot)
        print("left foot", L_foot)
                

        COM_x = 0 
        if R_ear[0] == 0 and L_ear[0] == 0:
            print("Error- head")
        elif R_ear[0] == 0:
            COM_x += L_ear[0] * body_perc["head"]
        else:
            COM_x += R_ear[0] * body_perc["head"]

        if body[0] == 0:
            print("Error- body")
        else:
            COM_x += body[0] * body_perc["body"]

        if pelvis[0] == 0:
            print("Error- pelvis")
        else:
            COM_x += pelvis[0]* body_perc["pelvis"]

        if R_arm[0] == 0 or L_arm[0] == 0:
            arms = max(R_arm[0],L_arm[0])
            if arms == 0:
                print("Error- arm")
            else:
                COM_x += arms * body_perc["arm"] * 2
        else:
            COM_x += (R_arm[0] + L_arm[0]) * body_perc["arm"]

        
        if R_forearm[0] == 0 or (L_forearm[0]) == 0:
            forearms = max(R_forearm[0],L_forearm[0])
            if forearms == 0:
                print("Error- forearm")
            else:
                COM_x += forearms * body_perc["forearm"] * 2
        else:
            print("4")
            COM_x += (R_forearm[0] + L_forearm[0]) * body_perc["forearm"]

        if R_hand[0] == 0 or L_hand[0] == 0:
            hands = max(R_hand[0],L_hand[0])
            if hands == 0:
                print("Error- arm")
            else:
                COM_x += hands * body_perc["hand"] * 2
        else:
          COM_x += (R_hand[0] + L_hand[0]) * body_perc["hand"]
        
        if R_thigh[0] == 0 or L_thigh[0] == 0:
            thighs = max(R_thigh[0],L_thigh[0])
            if thighs == 0:
                print("Error- thigh")
            else:
                COM_x += thighs * body_perc["thigh"] * 2
        else:
          COM_x += (R_thigh[0] + L_thigh[0]) * body_perc["thigh"]

        if R_shank[0] == 0 or L_shank[0] == 0:
            shanks = max(R_shank[0],L_shank[0])
            if shanks == 0:
                print("Error- shank")
            else:
                COM_x += shanks * body_perc["shank"] * 2
        else:
          COM_x += (R_shank[0] + L_shank[0]) * body_perc["shank"]

        if R_foot[0] == 0 or L_foot[0] == 0:
            foots = max(R_foot[0],L_foot[0])
            if foots == 0:
                print("Error- foot")
            else:
                COM_x += foots * body_perc["foot"] * 2
        else:
          COM_x += (R_foot[0] + L_foot[0]) * body_perc["foot"]

        COM_y = 0 
        if R_ear[1] == 0 and L_ear[1] == 0:
            print("Error- head")
        elif R_ear[1] == 0:
            COM_x += L_ear[1] * body_perc["head"]
        else:
            COM_x += R_ear[1] * body_perc["head"]

        if body[1] == 0:
          print("Error- body")
        else:
          COM_y += body[1]* body_perc["body"]

        if pelvis[1] == 0:
          print("Error- pelvis")
        else:
          COM_y += pelvis[1]* body_perc["pelvis"]

        if R_arm[1] == 0 or L_arm[1] == 0:
          arms = max(R_arm[1],L_arm[1])
          if arms == 0:
            print("Error- arm")
          else:
            COM_y += arms * body_perc["arm"] * 2
        else:
          COM_y += (R_arm[1] + L_arm[1]) * body_perc["arm"]

        if R_forearm[1] == 0 or L_forearm[1] == 0:
          forearms = max(R_forearm[1],L_forearm[1])
          if forearms == 0:
            print("Error- forearm")
          else:
            COM_y += forearms * body_perc["forearm"] * 2
        else:
          COM_y += (R_forearm[1] + L_forearm[1]) * body_perc["forearm"]

        if R_hand[1] == 0 or L_hand[1] == 0:
          hands = max(R_hand[1],L_hand[1])
          if hands == 0:
            print("Error- hand")
          else:
            COM_y += hands * body_perc["hand"] * 2
        else:
          COM_y += (R_hand[1] + L_hand[1]) * body_perc["hand"]
        
        if R_thigh[1] == 0 or L_thigh[1] == 0:
          thighs = max(R_thigh[1],L_thigh[1])
          if thighs == 0:
            print("Error- thigh")
          else:
            COM_y += thighs * body_perc["thigh"] * 2
        else:
          COM_y += (R_thigh[1] + L_thigh[1]) * body_perc["thigh"]

        if R_shank[1] == 0 or L_shank[1] == 0:
          shanks = max(R_shank[1],L_shank[1])
          if shanks == 0:
            print("Error- shank")
          else:
            COM_y += shanks * body_perc["shank"] * 2
        else:
          COM_y += (R_shank[1] + L_shank[1]) * body_perc["shank"]

        if R_foot[1] == 0 or L_foot[1] == 0:
          foots = max(R_foot[1],L_foot[1])
          if foots == 0:
            print("Error- foot")
          else:
            COM_y += foots * body_perc["foot"] * 2
        else:
          COM_y += (R_foot[1] + L_foot[1]) * body_perc["foot"]
        # COM_y = \
        #     head[1]*0.05 + \
        #     neck[1]*0.03 + \
        #     body[1]*0.27 + \
        #     pelvis[1]*0.11 + \
        #     shoulder[1]*0.05*2 + \
        #     arm[1]*0.03*2 + \
        #     forearm[1]*0.02*2 + \
        #     hand[1]*0.01*2 + \
        #     thigh[1]*0.1*2 + \
        #     shank[1]*0.04*2 + \
        #     foot[1]*0.02*2
        print(COM_x)
        print(COM_y)

        COM = (int(COM_x), int(COM_y))
        radius = 10
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