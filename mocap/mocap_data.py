import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import scipy.signal
import scipy.stats

def fill_data(value, time):
    """Fills/interpolates data
    
    Values for each missing millisecond are interpolated and returned in a new list

    args:
        value: values of data
        time: time values in ms of each result in value
    returns:
        new interpolated values
        time values in ms of each result in new interpolated values
    """
    new_time = range(0, time[-1]) #list of time values for each ms data sampled for
    new_value = [] #list of values interpolated for each ms
    for i in range(len(time) - 1):
        time_diff = time[i+1] - time[i] #difference in time between each sample
        if time_diff == 0:
            print(time[i])
        value_diff = value[i+1] - value[i] #difference in value between each sample
        value_ms = value_diff / time_diff #interpolated difference between each ms
        for j in range(time_diff):
            new_value.append(value[i] + (value_ms * j))
    return (new_value, new_time)

def gen_openpose_time(length, fps):
    """Time for openpose data
    
    Generates time in milliseconds for each openpose datapoint

    Args:
        len: length of openpose data
        fps: frame per second of video
    Returns:
        list: time in milliseconds
    """
    time = [math.floor(x * 1000 / fps) for x in range(length)]
    return time

def cut_data(data, start, length):
    """Cut data length
    
    Cut the length of the data to match with openpose data. Will reset time to 0 at cutpoint.

    Args:
        data: wii data in dictionary format: {time(ms): value}
        start: point to start cutting from
        length: number of data points to keep
    Returns:
        dict: cut data
    """
    try:
        cut = dict((k - start, data[k]) for k in range(start, start + length))
        return cut
    except:
        print("Error - not enough data to cut at point")

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

with open('mocap/brighton_mocap_ML_3.csv') as csv_file:
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
        if line_count > 1:
            time.append(int(float(row[1])*1000))
            for part in body_parts.keys():
                index = part_col.get(part)
                body_parts[part].append(row[index:index+3])
                #R_Arm.append(float(row[index:index+3]_)

        line_count += 1

#Make lists for body parts coordinates
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
    COM_x.append((ele1 - ele2) * 100) #*100 to convert to cm


# Process openpose data
with open("mocap/test.json") as f:
    data = json.load(f)
# Flip sign of data due to video recoding mirror image
# For side_landscape test need to -5 from -x since ankle was not centered
OP_COM = [-x for x in data['processed']['CoG_cm']]
OP_time = gen_openpose_time(len(OP_COM), data['processed']['fps'])
#TODO: Remove later- only here now because didnt filter in vid_pose for these ones
filtered_OP_COM = scipy.signal.savgol_filter(OP_COM, 51, 3)
resampled_OP_COM, resampled_OP_time = fill_data(filtered_OP_COM, OP_time)

resampled_data, resampled_time = fill_data(COM_x, time)

#Cross correlation to find shift of data
corr = np.correlate(resampled_OP_COM, resampled_data, "full")
shift = (len(resampled_data) - np.argmax(corr))
print(shift)
mocap_data = dict(zip(resampled_time, resampled_data)) # store time, value in dict
cut_mocap_data = cut_data(mocap_data, shift, len(resampled_OP_time)) # cut to match openpose size

#print(origin_x)
#print(COM_x)

#Plot data
plt.subplot(2,1,1)
plt.plot(cut_mocap_data.keys(), cut_mocap_data.values())
plt.plot(resampled_OP_time, resampled_OP_COM)
plt.legend(["Mocap", "Openpose"])
plt.title("COM_x")
plt.xlabel("Time (ms)")
plt.ylabel("COM X (cm)")

plt.subplot(2,1,2)
plt.title("CoM_x Mocap vs Openpose")
plt.scatter(cut_mocap_data.values(), resampled_OP_COM, label="stars", color="green",marker="*", s=1)
plt.xlabel("Mocap data")
plt.ylabel("Openpose data")

#axes = plt.gca()
#axes.set_ylim([-15,15])

mocap_COM = list(cut_mocap_data.values())
similarity = scipy.stats.pearsonr(mocap_COM, resampled_OP_COM)
print(similarity)

RMSE = math.sqrt(sum([(a - b)**2 for a, b in zip(mocap_COM, resampled_OP_COM)]
) / len(mocap_COM))
print(RMSE)

#Normalized RMSE
abs_val = list(map(abs, resampled_OP_COM)) 
print(RMSE/sum(abs_val))

plt.show()

#same as pearson
print(np.corrcoef(mocap_COM , resampled_OP_COM))

