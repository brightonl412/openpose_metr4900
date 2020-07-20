import csv
import matplotlib.pyplot as plt
import numpy as np

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

# print(COP_x)
#print(COP_y)
# print(time)

#Plot data
plt.plot(time, COP_y)
plt.xlabel("Time (ms)")
plt.ylabel("COP Y (cm)")
axes = plt.gca()
#axes.set_ylim([-15,15])
plt.show()

