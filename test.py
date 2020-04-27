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
        #params["number_people_max"] = 1
        #save data as json to folder
        #Find a better way to do this. Currently saves each frame as json
        params["write_json"] = "json_output"

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

        #Video location as a string
        vid_location = "media/front_landscape_2.mp4"
        #vid_location = "video.avi"
        cap = cv2.VideoCapture(vid_location)
        
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #Find Video Format
        video_type = vid_location.split(".")[-1]
        if video_type == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('media/output.mp4',fourcc,fps,(int(width),int(height)))
        elif video_type == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('media/output.avi',fourcc,fps,(int(width),int(height)))
        else:
            print("Video format not supported")
            sys.exit(-1)

        frame_num = 0
        sway_tot = 0
        unsuccessful_frames = 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if ret == True:
                # Process Image
                datum = op.Datum()
                imageToProcess = frame

                frame_num += 1
                cv2.putText(imageToProcess, str(frame_num), (100,100), font, 1, (255,255,255), 1)

                datum.cvInputData = imageToProcess

                opWrapper.emplaceAndPop([datum])

                # Display Image
                print("Body keypoints: \n" + str(datum.poseKeypoints))
                cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)

                #Get x difference/sway between nose[0]/neck[1] with midhip[8]
                #nose_x = datum.poseKeypoints[0][0][0]
                #neck_x = datum.poseKeypoints[0][1][0]
                #midhip_x = datum.poseKeypoints[0][8][0]
                # part not found
                # maybe change to confidence level == 0 because technically part could just be on left edge 
                # if (nose_x == 0 or neck_x == 0 or midhip_x == 0):
                #     unsuccessful_frames += 1
                #     print("A part not found- frame not used")
                # else:
                #     print("nose", nose_x)
                #     print("neck", neck_x)
                #     print("midhip", midhip_x)
                #     sway = abs(nose_x - midhip_x)
                #     print("frame sway", sway)
                #     sway_tot += sway
                
                # Save frame to output video
                out.write(datum.cvOutputData)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
       # sway_avg = sway_tot/(frame_num - unsuccessful_frames)
       # print("sway average", int(sway_avg))

        #prints model part numbers
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