import csv
import matplotlib.pyplot as plt
import numpy as np
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
        value_diff = value[i+1] - value[i] #difference in value between each sample
        value_ms = value_diff / time_diff #interpolated difference between each ms
        for j in range(time_diff):
            new_value.append(value[i] + (value_ms * j))
    return (new_value, new_time)

time = []  #in milliseconds
COP_x = [] #in cm
COP_y = [] #in cm

#Open CSV file to obtain data and place in lists
with open('wii/front.csv') as csv_file:
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


filtered = scipy.signal.savgol_filter(COP_y, 51, 3)

resampled_data, resampled_time = fill_data(COP_y, time)
resampled_filtered , _ = fill_data(filtered, time)

#Plot data
plt.subplot(2,1,1)
plt.plot(time, COP_y)
plt.title("COP_y raw")
plt.xlabel("Time (ms)")
plt.ylabel("COP Y (cm)")

plt.subplot(2,1,2)
plt.title("CoP_y interpolated")
plt.plot(resampled_time, resampled_data)
plt.xlabel("Time (ms)")
plt.ylabel("COP Y (cm)")
#axes = plt.gca()
#axes.set_ylim([-15,15])
#plt.show()

similarity = scipy.stats.pearsonr(resampled_data, resampled_filtered)
print(similarity)


