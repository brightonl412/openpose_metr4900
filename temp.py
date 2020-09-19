import vid_pose 

inputvid = "/home/brightonl412/Documents/openpose_metr4900/media/front_landscape_test.mp4"
model = "/home/brightonl412/Documents/openpose/models"
orientation = "front"
gender = "male"
height = 177
weight = 70
outputvid = "/home/brightonl412/Documents/openpose_metr4900/media/"
#file_name = "json_output/19-09-20-12:22:44.json"
file_name = "json_output/19-09-20-12:50:06.json"
vid_pose.generate_output(inputvid, model, orientation, gender, height, weight, outputvid, file_name)
