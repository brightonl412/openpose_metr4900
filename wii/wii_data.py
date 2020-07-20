import csv

time = []  #in milliseconds
COP_x = [] #in cm
COP_y = [] #in cm

with open('example.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    start_time = None
    for row in csv_reader:
        data = row[0].split()

        if line_count == 0:
            start_time = int(data[0])
        time.append(int(data[0]) - start_time)
        COP_x.append(data[5])
        COP_x.append(data[6])
        line_count += 1

print(COP_x)
print(COP_y)
print(time)