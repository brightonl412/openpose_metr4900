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

time = []  #in milliseconds
COP_x = [] #in cm
COP_y = [] #in cm

# front.json = front_landscape_test.csv
# front_l

#Open CSV file to obtain data and place in lists
with open('wii/brighton_wii_ML_2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    start_time = None
    for row in csv_reader:
        data = row[0].split()

        #Obtain initial start time
        if line_count == 0:
            start_time = int(data[0])
        
        time.append(int(data[0]) - start_time)
        COP_x.append(float(data[5]))
        COP_y.append(float(data[6]))
        line_count += 1


# Process openpose data
with open("wii/brighton_ML_2.json") as f:
    data = json.load(f)
# Flip sign of data due to video recoding mirror image
# For side_landscape test need to -5 from -x since ankle was not centered
OP_COP = [-x for x in data['processed']['CoP_cm']]
OP_time = gen_openpose_time(len(OP_COP), data['processed']['fps'])
#TODO: Remove later- only here now because didnt filter in vid_pose for these ones
filtered_OP_COP = scipy.signal.savgol_filter(OP_COP, 51, 3)
resampled_OP_COP, resampled_OP_time = fill_data(filtered_OP_COP, OP_time)

# Process wii data to match openpose 
filtered = scipy.signal.savgol_filter(COP_x, 51, 3)

resampled_data, resampled_time = fill_data(COP_x, time)
resampled_filtered , _ = fill_data(filtered, time)

#Cross correlation to find shift of data
corr = np.correlate(resampled_OP_COP, resampled_data, "full")
shift = (len(resampled_filtered) - np.argmax(corr))
wii_data = dict(zip(resampled_time, resampled_filtered)) # store time, value in dict
cut_wii_data = cut_data(wii_data, shift, len(resampled_OP_time)) # cut to match openpose size

#Plot data
plt.subplot(2,1,1)
plt.plot(cut_wii_data.keys(), cut_wii_data.values())
plt.plot(resampled_OP_time, resampled_OP_COP)
plt.legend(["Wii", "Openpose"])
plt.title("COP_x")
plt.xlabel("Time (ms)")
plt.ylabel("COP X (cm)")

plt.subplot(2,1,2)
plt.title("CoP_x Wii vs Openpose")
plt.scatter(cut_wii_data.values(), resampled_OP_COP, label="stars", color="green",marker="*", s=1)
plt.xlabel("Wii data")
plt.ylabel("Openpose data")

#axes = plt.gca()
#axes.set_ylim([-15,15])

wii_COP = list(cut_wii_data.values())
similarity = scipy.stats.pearsonr(wii_COP, resampled_OP_COP)
print(similarity)

RMSE = math.sqrt(sum([(a_i - b_i)**2 for a_i, b_i in zip(wii_COP, resampled_OP_COP)]
) / len(wii_COP))
print(RMSE)

# cross correlation 
# corr = np.correlate(wii_COP, resampled_OP_COP, "full")
# plt.subplot(2,1,2)
# lag = np.argmax(corr)-corr.size/2
# print(lag)
# plt.plot(corr)
plt.show()


