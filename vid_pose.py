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
    print('Error: OpenPose library could not be found. Did you enable \
        `BUILD_PYTHON` in CMake and have this Python script in the right \
        folder?')
    raise e
#import openpose




def set_gender(gender):
    """Sets body pecentages based on gender

    The percentages of body parts are set depending on gender. 

    Args:
        gender: str- must be "male" or "female"
    Returns: dict- body mass percentages
    """
    if gender == "male":
        return {
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
    elif gender == "female":
        return {
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
    else:
        print("Not a valid gender")
        sys.exit(-1)

#Maybe need to average values
#May need to chnage to be pos(i)- pos(i-step_size), change range as well
def calc_vel(position, step_size):
    """Calculate velocity for all possible frames

    Velocity calculation dependent upon position values using formula: 
    change in displacement/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames

    Returns: list- velocities of each frame starting from the step_size
    """
    com_vel = []
    for i in range(step_size + 1, len(position)):
        com_vel.append(calc_vel_frame(position, step_size, i))
    return com_vel

def calc_vel_frame(position, step_size, frame):
    """Calculate velocity for single fram

    Velocity calculation dependent upon position values using formula: 
    change in displacement/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames


    Returns: list- velocities of each frame starting from the step_size
    """
    if (len(position) < step_size or len(position) < frame):
        print("error")
    if (frame < step_size):
        print("error")
    
    vel = (position[frame] - position[frame - step_size]) / step_size
    return vel
    

#Need to complete
def calc_avg_vel(position, avg_quantity, step_size):
    """Calculate velocity using the average of a subset of frames

    Velocity calculation dependent upon position values using formula: 
    average change in displacement/change in time

    Args:
        postion: list- positions per frame
        avg_quantity- the number of frames to average
        step_size: int- change in time/frames

    Returns: list- velocities of each frame starting from the step_size
    """
    com_vel = []
    for i in range(0, len(position) - step_size):
        vel = (position[i + step_size] - position[i])/step_size
        com_vel.append(vel)
    return com_vel

def calc_acc(velocity, step_size):
    """Calculate acceleration

    Acceleration calculation dependent upon velocity values using formula: 
    change in velocity/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames

    Returns: list- acceleration of each frame starting from the step_size + \
        step_size of calc_vel
    """
    com_acc = []
    for i in range(0, len(velocity) - step_size):
        acc = (velocity[i + step_size] - velocity[i])/step_size
        com_acc.append(acc)
    return com_acc

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

        #Set gender of patient
        body_perc = set_gender("male")

        #Video location as a string
        vid_location = "media/landscape_1.mp4"
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
            out = cv2.VideoWriter('media/output.mp4', fourcc, fps, (int(width),
                int(height)))
        elif video_type == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('media/output.avi', fourcc, fps, (int(width),
                int(height)))
        else:
            print("Video format not supported")
            sys.exit(-1)

        frame_num = 0
        unsuccessful_frames = 0

        com_x_pos = []
        com_y_pos = []

        while(cap.isOpened()):
            ret, frame = cap.read()
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if ret == True:
                # Process Image
                datum = op.Datum()
                imageToProcess = frame

                frame_num += 1
                cv2.putText(imageToProcess, str(frame_num), (100,100), font, 1, 
                    (255,255,255), 1)
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])
                
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

                COM = (int(COM_x), int(COM_y))
                com_x_pos.append(int(COM_x))
                
                radius = 10
                # Blue color in BGR 
                color = (255, 0, 0) 
                thickness = 2
                #Add COM circle to image
                output_frame = datum.cvOutputData
                cv2.circle(output_frame, COM, radius, color, thickness)
           
                cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", output_frame)
                # Save frame to output video
                out.write(output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows

        vel = calc_vel(com_x_pos, 5)
        acc = calc_acc(vel, 5)
        print(vel)
        print(acc)

        test = calc_vel_frame(com_x_pos, 5, 6)
        print(test)

    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == '__main__':
    main()
