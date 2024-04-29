import numpy as np
import serial
import time
import datetime
import os

ct = datetime.datetime.now()

foldername = str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)+"_"+str(ct.hour)+"_"+str(ct.minute)+"_"+str(ct.second)
print(foldername)

path = "Intel_data/"+foldername+"/"
os.mkdir("Intel_data/"+foldername)
os.mkdir("Intel_data/"+foldername+"/images")
os.mkdir("Intel_data/"+foldername+"/pointcloud")

s = serial.Serial('COM4')

ser_arr = []
start_time = time.time()
time_diff = 0

while time_diff<5:
    res = s.readline()
    # print(res.decode("utf-8"))
    res = res.decode("utf-8")
    res = res.rstrip()
    ser_arr.append(res)
    current_time = time.time()
    time_diff = current_time - start_time
    # print(time_diff)

np.savetxt(path+"lidar.csv", ser_arr, delimiter=", ", fmt="%s")
