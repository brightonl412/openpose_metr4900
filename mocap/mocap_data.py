import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import math

#TODO: maybe change to dict of lists
R_Arm = []
body_parts = {}

#Male body %
body_perc = {
    "head": 0.0694,
    "upper_trunk": 0.1596,
    "mid_trunk": 0.1633,
    "lower_trunk": 0.1117,
    "arm": 0.0271,
    "forearm": 0.0162,
    "hand": 0.0061,
    "thigh": 0.1416,
    "shank": 0.0433,
    "foot": 0.0137
}

time = []

with open('mocap/test_1_mocap.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    start_time = None
    part_col = {}
    for row in csv_reader:

        #Obtain part labels and their colum index
        if line_count == 0:
            print(row)
            for i in range(2, len(row), 3):
                part_col[row[i]] = i
                body_parts[row[i]] = []
                
        if line_count > 2:
            time.append(row[1])
            for part in body_parts.keys():
                index = part_col.get(part)
                body_parts[part].append(row[index:index+3])
                #R_Arm.append(float(row[index:index+3]_)

        line_count += 1
#print(body_parts["Head"])
head_x = [float(x[2]) for x in body_parts["Head"]]
lower_trunk_x = [float(x[2]) for x in body_parts["Lower_Trunk"]]
mid_trunk_x = [float(x[2]) for x in body_parts["Mid_Trunk"]]
upper_trunk_x = [float(x[2]) for x in body_parts["Upper_Trunk"]]
l_arm_x = [float(x[2]) for x in body_parts["L_Arm"]]
r_arm_x = [float(x[2]) for x in body_parts["R_Arm"]]
l_forearm_x = [float(x[2]) for x in body_parts["L_Forearm"]]
r_forearm_x = [float(x[2]) for x in body_parts["R_Forearm"]]
l_hand_x =[float(x[2]) for x in body_parts["L_Hand"]]
r_hand_x =[float(x[2]) for x in body_parts["R_Hand"]]
l_thigh_x = [float(x[2]) for x in body_parts["L_Thigh"]]
r_thigh_x = [float(x[2]) for x in body_parts["R_Thigh"]]
l_shank_x = [float(x[2]) for x in body_parts["L_Shank"]]
r_shank_x = [float(x[2]) for x in body_parts["R_Shank"]]
l_foot_x = [float(x[2]) for x in body_parts["L_Foot"]]
r_foot_x = [float(x[2]) for x in body_parts["R_Foot"]]

# Obtain origin as center between two feet
origin_x = [(g + h) / 2 for g, h in zip(l_foot_x, r_foot_x)]

COM_x_3d = []
for i in range(len(origin_x)):
    COM_x_frame = 0
    COM_x_frame += head_x[i] * body_perc["head"]
    COM_x_frame += lower_trunk_x[i] * body_perc["lower_trunk"]
    COM_x_frame += mid_trunk_x[i] * body_perc["mid_trunk"]
    COM_x_frame += upper_trunk_x[i] * body_perc["upper_trunk"]
    COM_x_frame += l_arm_x[i] * body_perc["arm"]
    COM_x_frame += r_arm_x[i] * body_perc["arm"]
    COM_x_frame += l_forearm_x[i] * body_perc["forearm"]
    COM_x_frame += r_forearm_x[i] * body_perc["forearm"]
    COM_x_frame += l_hand_x[i] * body_perc["hand"]
    COM_x_frame += r_hand_x[i] * body_perc["hand"]
    COM_x_frame += l_thigh_x[i] * body_perc["thigh"]
    COM_x_frame += r_thigh_x[i] * body_perc["thigh"]
    COM_x_frame += l_shank_x[i] * body_perc["shank"]
    COM_x_frame += r_shank_x[i] * body_perc["shank"]
    COM_x_frame += l_foot_x[i] * body_perc["foot"]
    COM_x_frame += r_foot_x[i] * body_perc["foot"]

    COM_x_3d.append(COM_x_frame)
COM_x = []
for ele1, ele2 in zip(COM_x_3d, origin_x):
    COM_x.append(ele1 - ele2)
#print(origin_x)
#print(COM_x)

#Plot data
plt.plot(time, COM_x)
plt.title("COM_x")
plt.xlabel("Time (ms)")
plt.ylabel("COM X (cm)")
plt.show()