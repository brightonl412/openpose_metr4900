import math

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

    Returns: list- velocities of each frame starting from the step_size
    """
    avg_disp = int(math.floor(avg_quantity/2))
    start_frame = step_size + avg_disp + 1
    end_frame = len(position) - avg_disp
    print("Calculating velocities from frames", start_frame, "to", end_frame)
    com_vel = []
    for i in range(start_frame + 1, len(position) - avg_disp):
        com_vel.append(calc_avg_vel_frame(position, step_size, i, avg_quantity))
    return com_vel

def calc_avg_vel_frame(position, step_size, frame, avg_quantity):
    """Calculate velocity for a single frame using the averaged position

    Velocity calculation dependent upon position values using formula: 
    change in displacement/change in time

    Args:
        postion: list- positions per frame
        step_size: int- change in time/frames


    Returns: velocity of chosen frame 
    """
    avg_disp = int(math.floor(avg_quantity/2))

    if (frame < (step_size + avg_disp)):
        raise IndexError("Can not calculate for this frame")
    else:
        try:
            position_avg = 0
            for i in range(frame - 1 - avg_disp, frame + avg_disp):
                position_avg += position[i]
            position_1 = position_avg / (avg_disp * 2 + 1)
            
            position_avg = 0
            for i in range(frame - 1 - avg_disp- step_size, frame + avg_disp - step_size):
                position_avg += position[i]
            position_2 = position_avg / (avg_disp * 2 + 1)

            vel = (position_1 - position_2) / step_size
            return round(vel, 2)
        except IndexError:
            print("Frame or step_size out of bounds")


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
    com_x_pos = [1053, 1053, 1053, 1060, 1058, 1058, 1058, 1059, 1059, 1059, 1059, 1058, 1049, 1054, 1046, 1053, 1054, 1046, 1046, 1053, 1052, 1051, 1053, 1053, 1048, 1059, 1050, 1050, 1047, 1055, 1058, 1058, 1058, 1059, 1051, 1052, 1062, 1054, 1055, 1055, 1055, 1054, 1055, 1057, 1064, 1059, 1067, 1067, 1070, 1064, 1067, 1070, 1072, 1075, 1084, 1087, 1089, 1092, 1088, 1100, 1095, 1096, 1105, 1108, 1110, 1111, 1103, 1112, 1113, 1114, 1112, 1118, 1120, 1128, 1115, 1116, 1131, 1117, 1118, 1108, 1118, 1119, 1117, 1134, 1133, 1133, 1133, 1135, 1134, 1136, 1134, 1138, 1136, 1133, 1132, 1133, 1132, 1132, 1132, 1131, 1129, 1130, 1129, 1127, 1112, 1111, 1110, 1109, 1107, 1104, 1102, 1099, 1093, 1093, 1089, 1084, 1081, 1078, 1076, 1072, 1070, 1071, 1068, 1066, 1068, 1067, 1068, 1067, 1065, 1062, 1065, 1065, 1065, 1069, 1068, 1066, 1067, 1068, 1068, 1068, 1069, 1068, 1067, 1058, 1057, 1063, 1065, 1055, 1055, 1064, 1061, 1058, 1055, 1056, 1054]
    v = calc_avg_vel_frame(com_x_pos, 5, 8, 5)

    print(v)
if __name__ == '__main__':
    main()