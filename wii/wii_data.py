import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
import json
import math

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
        value_diff = value[i+1] - value[i] #difference in value between each sample
        value_ms = value_diff / time_diff #interpolated difference between each ms
        for j in range(time_diff):
            new_value.append(value[i] + (value_ms * j))
    return (new_value, new_time)

def gen_openpose_time(len, fps):
    """Time for openpose data
    
    Generates time in milliseconds for each openpose datapoint

    Args:
        len: length of openpose data
        fps: frame per second of video
    Returns:
        list: time in milliseconds
    """
    time = [math.floor(x * 1000 / fps) for x in range(len)]
    return time

time = []  #in milliseconds
COP_x = [] #in cm
COP_y = [] #in cm

#Open CSV file to obtain data and place in lists
with open('wii/front_landscape_test.csv') as csv_file:
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


filtered = scipy.signal.savgol_filter(COP_x, 51, 3)

resampled_data, resampled_time = fill_data(COP_x, time)
resampled_filtered , _ = fill_data(filtered, time)

# Process openpose data
with open("wii/front.json") as f:
    data = json.load(f)
# Flip sign of data due to video recoding mirror image
OP_COP = [-x for x in data['processed']['CoP_cm']]
OP_time = gen_openpose_time(len(OP_COP), data['processed']['fps'])


#Plot data
plt.subplot(2,1,1)
plt.plot(time, COP_x)
plt.title("COP_x raw")
plt.xlabel("Time (ms)")
plt.ylabel("COP X (cm)")

plt.subplot(2,1,2)
plt.title("CoP_x interpolated")
plt.plot(OP_time, OP_COP)
plt.xlabel("Time (ms)")
plt.ylabel("COP X (cm)")
#axes = plt.gca()
#axes.set_ylim([-15,15])
plt.show()

similarity = scipy.stats.pearsonr(resampled_data, resampled_filtered)
print(similarity)


