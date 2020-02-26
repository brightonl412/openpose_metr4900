import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

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
        #parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        #parser.add_argument("--image_dir", default="../openpose/examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        #params["model_folder"] = "../../../models/"
        params["model_folder"] = "../openpose/models/"
        params["number_people_max"] = 1

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


        cap = cv2.VideoCapture('video.avi')
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        #Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc,20.0,(1280,720))
        font = cv2.FONT_HERSHEY_SIMPLEX

        frameNumber = 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if ret == True:

                # Process Image
                datum = op.Datum()
                imageToProcess = frame

                
                frameNumber = frameNumber + 1
                cv2.putText(imageToProcess, str(frameNumber), (100,100), font, 1, (255,255,255), 1)

                datum.cvInputData = imageToProcess

                opWrapper.emplaceAndPop([datum])

                # Display Image
                print("Body keypoints: \n" + str(datum.poseKeypoints))
                cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)

                # Save frame to output video
                out.write(datum.cvOutputData)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        poseModel = op.PoseModel.BODY_25
        print(op.getPoseBodyPartMapping(poseModel))
        cap.release()
        out.release()
        cv2.destroyAllWindows

    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == '__main__':
    main()
