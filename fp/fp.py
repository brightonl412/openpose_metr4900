import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
import json
import math
from sklearn.metrics import mean_squared_error

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
            print(i)
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
        data: fp data in dictionary format: {time(ms): value}
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

time = []  #in milliseconds
COP_x = [] #in cm
COP_y = [] #in cm

x_center = -0.2492585
y_center = 0.572417


body_parts = {}
with open('fp/mocap/S03_MOCAP_005.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    part_col = {}
    for row in csv_reader:
        #Obtain part labels and their colum index
        if line_count == 3:
            for i in range(2, len(row), 3):
                part_col[row[i]] = i
                body_parts[row[i]] = []
        if line_count == 7:
            for part in body_parts.keys():
                index = part_col.get(part)
                body_parts[part].append(row[index:index+3])
            break
        line_count += 1
l_ankle = float(body_parts['L_Ankle'][0][2])
r_ankle = float(body_parts['R_Ankle'][0][2])
pend_origin = (l_ankle + r_ankle)/2
print(pend_origin)
offset = pend_origin - y_center
print(offset)

# front.json = front_landscape_test.csv
# front_l

#Open CSV file to obtain data and place in lists
with open('fp/fp_data/S03AP005.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count % 3 == 0:
            time.append(int(float(row[0])*1000))
            COP_x.append(float(row[1]))
            COP_y.append(float(row[2]))
        line_count += 1
COP_x = [(x + (100*offset)) for x in COP_x]

# Process fp data to match openpose 
filtered = scipy.signal.savgol_filter(COP_x, 51, 3)

resampled_data, resampled_time = fill_data(COP_x, time)
resampled_filtered , _ = fill_data(filtered, time)

# Process openpose data
with open("fp/op_data/S03AP005.json") as f:
    data = json.load(f)
# Flip sign of data due to video recoding mirror image
# For side_landscape test need to -5 from -x since ankle was not centered
OP_COP = [x for x in data['processed']['CoP_cm']]
OP_time = gen_openpose_time(len(OP_COP), data['processed']['fps'])
#TODO: Remove later- only here now because didnt filter in vid_pose for these ones
filtered_OP_COP = scipy.signal.savgol_filter(OP_COP, 51, 3)
resampled_OP_COP, resampled_OP_time = fill_data(filtered_OP_COP, OP_time)

#Cross correlation to find shift of data
corr = np.correlate(resampled_OP_COP, resampled_filtered, "full")
shift = (len(resampled_filtered) - np.argmax(corr))
fp_data = dict(zip(resampled_time, resampled_filtered)) # store time, value in dict
cut_fp_data = cut_data(fp_data, shift, len(resampled_OP_time)) # cut to match openpose size

#Plot data
plt.subplot(2,1,1)
plt.plot(cut_fp_data.keys(), cut_fp_data.values())
plt.plot(resampled_OP_time, resampled_OP_COP)
#plt.plot(resampled_time, resampled_data)
plt.legend(["Force Plate", "OpenPose"])
plt.title("CoP")
plt.xlabel("Time (ms)")
plt.ylabel("COP (cm)")

plt.subplot(2,1,2)
plt.title("CoP FP vs Openpose")
plt.scatter(cut_fp_data.values(), resampled_OP_COP, label="stars", color="green",marker="*", s=1)
plt.xlabel("FP data (cm)")
plt.ylabel("OpenPose data (cm)")

plt.subplots_adjust(hspace=0.5)

#axes = plt.gca()
#axes.set_ylim([-15,15])

fp_COP = list(cut_fp_data.values())
similarity = scipy.stats.pearsonr(fp_COP, resampled_OP_COP)
print(similarity)

RMSE = math.sqrt(sum([(a - b)**2 for a, b in zip(fp_COP, resampled_OP_COP)]
) / len(fp_COP))
print(RMSE)

#Normalized RMSE
abs_val = list(map(abs, resampled_OP_COP)) 
print(RMSE/sum(abs_val))

# cross correlation 
# corr = np.correlate(fp_COP, resampled_OP_COP, "full")
# plt.subplot(2,1,2)
# lag = np.argmax(corr)-corr.size/2
# print(lag)
# plt.plot(corr)
plt.show()


