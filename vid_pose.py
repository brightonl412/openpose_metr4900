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

dir_path = os.path.dirname(os.path.realpath(__file__))

try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable \
        `BUILD_PYTHON` in CMake and have this Python script in the right \
        folder?')
    raise e

def calc_vel(position, step_size):
    """Calculate velocity for all possible frames

    Velocity calculation dependent upon position values using formula: 
    change in displacement/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames

    Returns:
        int- starting frame number
        int- ending frame number
        list- velocities of each frame starting from the start frame
    """
    start_frame = step_size + 1
    end_frame = len(position)
    com_vel = []
    for i in range(start_frame, end_frame + 1):
        com_vel.append(calc_vel_frame(position, step_size, i))
    return start_frame, end_frame, com_vel

def calc_vel_frame(position, step_size, frame):
    """Calculate velocity for single frame

    Velocity calculation dependent upon position values using formula: 
    change in displacement/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames

    Returns: velocity of chosen frame 
    """
    if (frame < step_size):
        raise IndexError("Frame must be greater than step size")
    else:
        try:
            vel = (position[frame - 1] - position[frame - 1 - step_size]) / step_size
            return vel
        except IndexError:
            print("Frame or step_size out of bounds")
    

def calc_avg_vel(position, step_size, avg_quantity):
    """Calculate velocity using the average of a subset of frames

    Velocity calculation dependent upon position values using formula: 
    average change in displacement/change in time

    Args:
        postion: list- positions per frame
        avg_quantity- the number of frames to average
        step_size: int- change in time/frames

    Returns: 
        int- starting frame number
        int- ending frame number
        list- velocities of each frame starting from the start frame
    """
    avg_disp = int(math.floor(avg_quantity / 2))
    start_frame = step_size + avg_disp + 1
    end_frame = len(position) - avg_disp
    print("Calculating velocities from frames", start_frame, "to", end_frame)
    com_vel = []
    for i in range(start_frame, end_frame + 1):
        com_vel.append(calc_avg_vel_frame(position, step_size, i, avg_quantity))
    return start_frame, end_frame, com_vel

def calc_avg_vel_frame(position, step_size, frame, avg_quantity):
    """Calculate velocity for a single frame using the averaged position

    Velocity calculation dependent upon position values using formula: 
    change in displacement/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames

    Returns: velocity of chosen frame 
    """
    avg_disp = int(math.floor(avg_quantity / 2))

    if (frame < (step_size + avg_disp)):
        raise IndexError("Can not calculate for this frame")
    else:
        try:
            position_avg = 0
            for i in range(frame - 1 - avg_disp, frame + avg_disp):
                position_avg += position[i]
            position_1 = position_avg / (avg_disp * 2 + 1)
            
            position_avg = 0
            for i in range(frame - 1 - avg_disp - step_size, frame + avg_disp - step_size):
                position_avg += position[i]
            position_2 = position_avg / (avg_disp * 2 + 1)

            vel = (position_1 - position_2) / step_size
            return vel
            #return round(vel, 2)
        except IndexError:
            print("Frame or step_size out of bounds")

def calc_acc(velocity, step_size, vel_start_frame):
    """Calculate acceleration

    Acceleration calculation dependent upon velocity values using formula: 
    change in velocity/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames
        vel_start_frame: int- the frame number of the first index in velocity 

    Returns: 
        int- starting frame number
        int- ending frame number
        list- velocities of each frame starting from the start frame
    """
    start_frame = step_size + vel_start_frame
    end_frame = len(velocity) + vel_start_frame - 1
    com_acc = []
    print("Calculating acceleration from frames", start_frame, "to", end_frame)
    for i in range(start_frame, end_frame + 1):
        com_acc.append(calc_acc_frame(velocity, step_size, i, vel_start_frame))
    return start_frame, end_frame, com_acc

def calc_acc_frame(velocity, step_size, frame, vel_start_frame):
    """Calculate acceleration for single frame

    Acceleration calculation dependent upon position values using formula: 
    change in velocity/change in time

    Args:
        velocity: list- velocity of CoM per frame
        step_size: int- change in time- number of frames

    Returns: acceleration of chosen frame 
    """
    #The offset required due to the velocities starting a vel_start_frame
    acc_offset = frame - vel_start_frame + 1
    if ((acc_offset) < step_size):
        raise IndexError("Acceleration cannot be calculated for this frame")
    else:
        try:
            acc = (velocity[acc_offset - 1] - velocity[acc_offset - 1 - step_size]) / step_size
            return acc
            #return round(acc,2)
        except IndexError:
            print("Frame or step_size out of bounds")

def calc_inertia(CoM, pend_origin, mass):
    """Calculate mass moment of inertia

    Mass moment of inertia calculation for the patient. Given by formula: Mr^2,
    where:
        M- Mass
        r- radius of mass from axis of rotation (pend_origin)
    The mass moment of inertia is given in kg*pixels^2

    Args:
        CoM: list- x and y Centre of Mass coordinates in pixels
        pend_origin: list- x and y coorindates of pend_origin in pixels
        mass: int- mass of patient in kg

    Returns: Mass moment of inertia 
    """
    dist = math.sqrt(((CoM[0] - pend_origin[0])**2) + 
        ((CoM[1] - pend_origin[1])**2))
    inertia = mass * (dist**2)
    return inertia

def calc_force(patient, fps):
    """Calculate force on CoM and resultant force on CoP

    The force calculated is the force due to gravity on the patient's mass. This
    force is converted to units: kg*pixels/frame^2

    Args:
        Patient: patient object
        fps:     float- frames per second of the video
    
    Returns: Force due to CoM
    """
    gravity = 9.81 # m/s^2
    mass = patient.mass
    m_to_pixel = patient.pixel_cm * 100
    acceleration = gravity * m_to_pixel / (fps**2) # pixels/frame^2
    force = mass * acceleration #kg pixels/frame^2
    return force

def CoG_x(CoM_x, pend_origin):
    """Center of Gravity x position

    Calulates the x displacement of the CoM to the inverted pendulumn model
    origin. For a video in the A/P direction the origin will be the ankle, and 
    in the M/: direction the origin will be the mid point between the 2 ankles.

    Args:
        CoM_x:       list- positions of x CoM for all frames
        pend_origin: list- positions of the pendulum origin for all frames
    
    Returns: list- x direction CoG for all frames
    """
    CoG = []
    for i in range(0, len(CoM_x)):
        CoG_frame = CoM_x[i] - pend_origin[i][0]
        CoG.append(CoG_frame)
    return CoG

def CoP_x(CoG_x, ang_acc, inertia, force):
    """Center of Pressure x position

    Calulates the x displacement of the CoP to the inverted pendulumn model
    origin. For a video in the A/P direction the origin will be the ankle, and 
    in the M/: direction the origin will be the mid point between the 2 ankles.

    Args:
        CoG_x:   list- positions of x CoG for all frames
        ang_acc: list- angular acceleration of CoM for all frames
        inertia: list- mass moment of inertia for all frames
        force:   int- force due to gravity on CoM
    
    Returns: list- x direction CoP from the ang acc start frame
    """
    CoP = []
    for i in range(0, len(ang_acc)):
        if ang_acc[i] is None:
            pass
        else:
            CoP_frame = CoG_x[i] + (inertia[i] * ang_acc[i] / force)
            CoP.append(CoP_frame)
    return CoP

def length(v):
    """Computes length of line

    Args:
        v: list- x and y position of point from origin

    Returns: float- length of line in pixels
    """
    return math.sqrt(v[0]**2 + v[1]**2)

def dot_product(v,w):
    """Computes dot product between two lines

    Args:
        v: list- x and y position of point 1 from origin
        w: list- x and y position of point 2 from origin

    Returns: float- dot product in pixels
    """
    return v[0] * w[0] + v[1] * w[1]

def determinant(v,w):
    """Computes determinant of two lines

    Args:
        v: list- x and y position of point 1 from origin
        w: list- x and y position of point 2 from origin

    Returns: float- determinant
    """
    return v[0] * w[1] - v[1] * w[0]

def angle(v,w):
    """Computes angle between two lines

    Args:
        v: list- x and y position of point 1 from origin
        w: list- x and y position of point 2 from origin

    Returns: float- angle between the two lines in radians
    """
    cosx = dot_product(v,w) / (length(v) * length(w))
    #det = determinant(A,B)
    rad = math.acos(cosx) # in radians
    return rad
    #return rad*180/math.pi # returns degrees

#Add none to make all lists start at frame 1
def add_empty_frames(frames, start):
    """Add empty frames to list

    Used to add None to start of list so that displacement, velcity and
    acceleration have the same number of elements.

    Args:
        frames: list- list to added empty frames
        start:  int- the frame number which the list initially starts at
    
    Returns:
        list- the updated list with the empty frames added
    """
    updated = copy.copy(frames)
    for i in range(1, start):
        updated.insert(0, None)
    return updated

def generate_output(inputvid, model, orientation, gender, height, weight, outputvid):
    try:
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = model
        #Find a better way to do this. Currently saves each frame as json
        params["write_json"] = "json_output"

        #Configure Openpose Python Wrapper
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        #Create patient object of person in video
        patient = Patient(gender, height, weight)
        body_perc = patient.body_perc()

        vid_location = inputvid
        cap = cv2.VideoCapture(vid_location)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #Find Video Format
        video_type = vid_location.split(".")[-1]
        if video_type == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            location = outputvid + '/output.mp4'
            out = cv2.VideoWriter(location, fourcc, fps, (int(width),
                int(height)))
        elif video_type == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            location = outputvid + '/output.avi'
            out = cv2.VideoWriter(location, fourcc, fps, (int(width),
                int(height)))
        else:
            print("Video format not supported")
            sys.exit(-1)

        frame_num = 0

        #lists to store data per frame 
        com_x_pos = []
        com_y_pos = []
        com_ang = []
        inertias = []
        pend_origin = []

        print("Generating Pose")
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
                
                #limb positions
                body = [(datum.poseKeypoints[0][1][0] + datum.poseKeypoints[0][8][0])/2, (datum.poseKeypoints[0][1][1] +datum.poseKeypoints[0][8][1])/2]
                pelvis = [datum.poseKeypoints[0][8][0],datum.poseKeypoints[0][8][1]]

                R_ear = [datum.poseKeypoints[0][17][0],datum.poseKeypoints[0][17][1]]
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

                #Used for pixel-m conversion
                nose = [datum.poseKeypoints[0][0][0], datum.poseKeypoints[0][0][1]]
                
                foot_y = max(R_foot[1], L_foot[1])
                if foot_y == 0:
                    print("Error- Foot")
                nose_y = nose[1]
                if nose_y == 0:
                    print("Error- Nose")
                pixel_height = foot_y - nose_y
                patient.set_pixel_cm(pixel_height)

                
                COM_x = 0 
                COM_y = 0 
                if orientation == "side":
                    #Calulate x CoM
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
                        arms = max(R_arm[0], L_arm[0])
                        if arms == 0:
                            print("Error- arm")
                        else:
                            COM_x += arms * body_perc["arm"] * 2
                    else:
                        COM_x += (R_arm[0] + L_arm[0]) * body_perc["arm"]
                    
                    if R_forearm[0] == 0 or (L_forearm[0]) == 0:
                        forearms = max(R_forearm[0], L_forearm[0])
                        if forearms == 0:
                            print("Error- forearm")
                        else:
                            COM_x += forearms * body_perc["forearm"] * 2
                    else:
                        COM_x += (R_forearm[0] + L_forearm[0]) * body_perc["forearm"]

                    if R_hand[0] == 0 or L_hand[0] == 0:
                        hands = max(R_hand[0], L_hand[0])
                        if hands == 0:
                            print("Error- arm")
                        else:
                            COM_x += hands * body_perc["hand"] * 2
                    else:
                        COM_x += (R_hand[0] + L_hand[0]) * body_perc["hand"]
                    
                    if R_thigh[0] == 0 or L_thigh[0] == 0:
                        thighs = max(R_thigh[0], L_thigh[0])
                        if thighs == 0:
                            print("Error- thigh")
                        else:
                            COM_x += thighs * body_perc["thigh"] * 2
                    else:
                        COM_x += (R_thigh[0] + L_thigh[0]) * body_perc["thigh"]

                    if R_shank[0] == 0 or L_shank[0] == 0:
                        shanks = max(R_shank[0], L_shank[0])
                        if shanks == 0:
                            print("Error- shank")
                        else:
                            COM_x += shanks * body_perc["shank"] * 2
                    else:
                        COM_x += (R_shank[0] + L_shank[0]) * body_perc["shank"]

                    if R_foot[0] == 0 or L_foot[0] == 0:
                        foots = max(R_foot[0], L_foot[0])
                        if foots == 0:
                            print("Error- foot")
                        else:
                            COM_x += foots * body_perc["foot"] * 2
                    else:
                        COM_x += (R_foot[0] + L_foot[0]) * body_perc["foot"]

                    #Calulate y CoM
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
                        arms = max(R_arm[1], L_arm[1])
                        if arms == 0:
                            print("Error- arm")
                        else:
                            COM_y += arms * body_perc["arm"] * 2
                    else:
                        COM_y += (R_arm[1] + L_arm[1]) * body_perc["arm"]

                    if R_forearm[1] == 0 or L_forearm[1] == 0:
                        forearms = max(R_forearm[1], L_forearm[1])
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
                        thighs = max(R_thigh[1], L_thigh[1])
                        if thighs == 0:
                            print("Error- thigh")
                        else:
                            COM_y += thighs * body_perc["thigh"] * 2
                    else:
                        COM_y += (R_thigh[1] + L_thigh[1]) * body_perc["thigh"]

                    if R_shank[1] == 0 or L_shank[1] == 0:
                        shanks = max(R_shank[1], L_shank[1])
                        if shanks == 0:
                            print("Error- shank")
                        else:
                            COM_y += shanks * body_perc["shank"] * 2
                    else:
                        COM_y += (R_shank[1] + L_shank[1]) * body_perc["shank"]

                    if R_foot[1] == 0 or L_foot[1] == 0:
                        foots = max(R_foot[1], L_foot[1])
                        if foots == 0:
                            print("Error- foot")
                        else:
                            COM_y += foots * body_perc["foot"] * 2
                    else:
                        COM_y += (R_foot[1] + L_foot[1]) * body_perc["foot"]
                
                elif orientation == "front":
                    #Calulate x CoM
                    if nose[0] == 0 :
                        print("Error- head")
                    else:
                        COM_x += nose[0] * body_perc["head"]

                    if body[0] == 0:
                        print("Error- body")
                    else:
                        COM_x += body[0] * body_perc["body"]

                    if pelvis[0] == 0:
                        print("Error- pelvis")
                    else:
                        COM_x += pelvis[0]* body_perc["pelvis"]

                    if R_arm[0] == 0 or L_arm[0] == 0:
                        print("Error- arm")
                    COM_x += (R_arm[0] + L_arm[0]) * body_perc["arm"]
                    
                    if R_forearm[0] == 0 or (L_forearm[0]) == 0:
                        print("Error- forearm")
                    COM_x += (R_forearm[0] + L_forearm[0]) * body_perc["forearm"]

                    if R_hand[0] == 0 or L_hand[0] == 0:
                        print("Error- arm")
                    COM_x += (R_hand[0] + L_hand[0]) * body_perc["hand"]
                    
                    if R_thigh[0] == 0 or L_thigh[0] == 0:
                        print("Error- thigh")
                    COM_x += (R_thigh[0] + L_thigh[0]) * body_perc["thigh"]

                    if R_shank[0] == 0 or L_shank[0] == 0:
                        print("Error- shank")
                    COM_x += (R_shank[0] + L_shank[0]) * body_perc["shank"]

                    if R_foot[0] == 0 or L_foot[0] == 0:
                        print("Error- foot")
                    COM_x += (R_foot[0] + L_foot[0]) * body_perc["foot"]

                    #Calulate y CoM
                    if nose[1] == 0 :
                        print("Error- head")
                    else:
                        COM_y += nose[1] * body_perc["head"]

                    if body[1] == 0:
                        print("Error- body")
                    else:
                        COM_y += body[1]* body_perc["body"]

                    if pelvis[1] == 0:
                        print("Error- pelvis")
                    else:
                        COM_y += pelvis[1]* body_perc["pelvis"]

                    if R_arm[1] == 0 or L_arm[1] == 0:
                        print("Error- arm")
                    COM_y += (R_arm[1] + L_arm[1]) * body_perc["arm"]

                    if R_forearm[1] == 0 or L_forearm[1] == 0:
                        print("Error- forearm")
                    COM_y += (R_forearm[1] + L_forearm[1]) * body_perc["forearm"]

                    if R_hand[1] == 0 or L_hand[1] == 0:
                        print("Error- hand")
                    COM_y += (R_hand[1] + L_hand[1]) * body_perc["hand"]
                    
                    if R_thigh[1] == 0 or L_thigh[1] == 0:
                        print("Error- thigh")
                    COM_y += (R_thigh[1] + L_thigh[1]) * body_perc["thigh"]

                    if R_shank[1] == 0 or L_shank[1] == 0:
                        print("Error- shank")
                    COM_y += (R_shank[1] + L_shank[1]) * body_perc["shank"]

                    if R_foot[1] == 0 or L_foot[1] == 0:
                        print("Error- foot")
                    COM_y += (R_foot[1] + L_foot[1]) * body_perc["foot"]
                else:
                    print("Not a valid orientation")

                com_x_pos.append(int(COM_x))
                com_y_pos.append(int(COM_y))        
                
                #Mass Moment of Inertia Calc
                R_ankle = [datum.poseKeypoints[0][11][0],datum.poseKeypoints[0][11][1]]
                L_ankle = [datum.poseKeypoints[0][14][0],datum.poseKeypoints[0][14][1]]
                pend = None

                if orientation == "side":
                    if R_ankle[1] == 0 and L_ankle[1] == 0:
                        print("Error- ankle")
                    else:
                        if R_ankle[1] >= L_ankle[1]:
                            pend = R_ankle
                        else:
                            pend = L_ankle
                elif orientation == "front":
                    if R_ankle[1] == 0 or L_ankle[1] == 0:
                        print("Error- ankle")
                        #Test- use previous point in list
                        pend = pend_origin[-1]
                    else:
                        pend = [(L_ankle[0] + R_ankle [0]) / 2, 
                            (L_ankle[1] + R_ankle [1]) / 2]
                else:
                    print("Not a valid orientation")
                CoM = [int(COM_x), int(COM_y)]
                MMI = calc_inertia(CoM, pend, patient.mass)
                inertias.append(MMI)                    

                #Angle calculation
                horizontal_axis = [10, 0] #abitary point on the horizontal axis
                CoM_to_pend_origin = [COM_x - pend[0], pend[1]- COM_y]
                ang = angle(CoM_to_pend_origin, horizontal_axis)
                com_ang.append(ang) 
                pend_origin.append(pend)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        #out.release()
        cv2.destroyAllWindows

        print("Generating Output")
        cap = cv2.VideoCapture(vid_location)
        frame_num = 0
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #Apply Savistky-Golay Filter
        filtered_comx = scipy.signal.savgol_filter(com_x_pos, 51, 3)
        filtered_com_ang = scipy.signal.savgol_filter(com_ang, 51, 3)

        #Calculate frame velocities and accelerations
        start, stop, velocity = calc_vel(filtered_comx, 5)
        start2, stop2, acceleration = calc_acc(velocity, 5, start)

        #CoG, Force, Angular Acceleration
        ang_vel_start, _, ang_vel = calc_vel(filtered_com_ang, 5) #Angular vel
        ang_acc_start, ang_acc_stop, ang_acc = \
            calc_acc(ang_vel, 5, ang_vel_start) #Angular acc
        #Need the updated angular acceleration so that frames match with CoG
        updated_ang_acc = add_empty_frames(ang_acc, ang_acc_start)
        force = calc_force(patient, fps)
        CoG = CoG_x(filtered_comx, pend_origin)
        CoP = CoP_x(CoG, updated_ang_acc, inertias, force)


        #Graphs
        # x = np.linspace(1,len(com_x_pos), len(com_x_pos)) 
        # plt.subplot(3,1,1)
        # plt.scatter(x,com_x_pos,label="stars", color="green",marker="*", s=30)
        # plt.plot(x, filtered_comx, color = 'red')
        # plt.title("CoM x pos per Frame")
        # plt.xlabel("Frame")
        # plt.ylabel("CoM x pos (Pixels)")

        # plt.subplot(3,1,2)
        # x2 = np.linspace(start,stop, len(velocity)) 
        # plt.scatter(x2,velocity,label="stars", color="green",marker="*", s=30)
        # plt.title("Velocity per Frame")
        # plt.xlabel("Frame")
        # plt.ylabel("Velocity (Pixels/frame)")

        # plt.subplot(3,1,3)
        # x3 = np.linspace(start2,stop2, len(acceleration)) 
        # plt.scatter(x3,acceleration,label="stars", color="green",marker="*", s=30)
        # plt.title("Accerelation per Frame")
        # plt.xlabel("Frame")
        # plt.ylabel("Acceleration (Pixels^2/frame")

        # x = np.linspace(1,len(com_x_pos), len(com_x_pos)) 
        # plt.subplot(2,1,1)
        # plt.scatter(x,com_x_pos,label="stars", color="green",marker="*", s=30)
        # plt.title("CoM x pos per frame")
        # plt.xlabel("Frame")
        # plt.ylabel("CoM x pos (Pixels)")
        # plt.plot(x, filtered_comx, color = 'red')
        # plt.subplot(2,1,2)
        # plt.scatter(x,com_ang,label="stars", color="green",marker="*", s=30)
        # plt.title("CoM angle pos per frame")
        # plt.xlabel("Frame")
        # plt.ylabel("CoM ang pos (radians)")
        # plt.plot(x, filtered_com_ang, color = 'red')

        #plt.show()

        #Show each frame and save to output folder
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # Process Image
                datum = op.Datum()
                imageToProcess = frame
                frame_num += 1
                cv2.putText(imageToProcess, str(frame_num), (100,100), font, 1, 
                    (255,255,255), 1)
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])
                
                #Add COM circle to image
                radius = 10
                # Blue color in BGR 
                color = (255, 0, 0) 
                thickness = 2
                COM_x = int(filtered_comx[frame_num - 1])
                COM_y = com_y_pos[frame_num - 1]
                COM = (COM_x, COM_y)
                output_frame = datum.cvOutputData
                cv2.circle(output_frame, COM, radius, color, thickness)

                #Plot frame velocities
                if (frame_num >= start and 
                    frame_num <= stop):
                    vel = velocity[frame_num - start]
                    point_2 = (int(COM_x + 10 * vel), int(COM_y))
                    cv2.arrowedLine(output_frame, COM, point_2, (0,0,255), 3)

                #Plot CoP
                if (frame_num >= ang_acc_start and
                    frame_num <= ang_acc_stop):
                    CoP_frame = CoP[frame_num - ang_acc_start]
                    pend_origin_x = pend_origin[frame_num - 1][0]
                    pend_origin_y = pend_origin[frame_num - 1][1]
                    point_2 = (int(pend_origin_x + CoP_frame), int(pend_origin_y - 20)) 
                    point_1 = (int(pend_origin_x + CoP_frame), int(pend_origin_y))
                    cv2.arrowedLine(output_frame, point_1, point_2, (0,0,255), 3)

                cv2.imshow("Output video", output_frame)
                out.write(output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows
        
    except Exception as e:
        print(e)
        sys.exit(-1)

# if __name__ == '__main__':
#     main()
