import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import serial
import time
import csv

import trimesh
from zoedepth.utils.geometry import depth_to_points, create_triangles
from functools import partial
import tempfile
import PIL
from zoedepth.utils.misc import get_image_from_url, colorize
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from torchvision.transforms import ToTensor
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import mode



def predict_depth(model, image):
    depth_im = model.infer_pil(image)
    x = ToTensor()(image).unsqueeze(0).to(DEVICE)
    # depth = model.infer(x)
    return depth_im



# flags
loop_flag = True
showRS_flag = True
pc_flag = False
model_flag = False

if model_flag:
    conf = get_config("zoedepth", "infer")
    DEVICE = 'cpu'

    model = build_model(conf).to(DEVICE)
    # model.eval()




# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



ct = datetime.datetime.now()

foldername = str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)+"_"+str(ct.hour)+"_"+str(ct.minute)+"_"+str(ct.second)
print(foldername)

path = "Intel_data/"+foldername+"/"
os.mkdir("Intel_data/"+foldername)
os.mkdir("Intel_data/"+foldername+"/images")
os.mkdir("Intel_data/"+foldername+"/lidar")
os.mkdir("Intel_data/"+foldername+"/depthmap")
if pc_flag:
    os.mkdir("Intel_data/"+foldername+"/pointcloud")


mode_list = []
image_counter = 0


# Start streaming
profile = pipeline.start(config)
s = serial.Serial('COM4')

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

pc = rs.pointcloud()
points = rs.points()
colorizer = rs.colorizer()

align_to = rs.stream.color
align = rs.align(align_to)



while loop_flag:

    try:
        color = [0, 255, 0]
        centre = [320, 240]
        start_p = (centre[0]-100, centre[1]-100)
        end_p = (centre[0]+100, centre[1]+100)
        thickness = 2

        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            colorized = colorizer.process(frames)

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            color_rect = np.copy(color_image)
            color_rect = cv2.rectangle(color_rect, start_p, end_p, color, thickness)
            cv2.imshow('camera', color_rect)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                ser_arr = []
                start_time = time.time()
                time_diff = 0

                depth_image = np.asanyarray(depth_frame.get_data())

                aligned_frames = align.process(frames)
                
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                aligned_color_frame = aligned_frames.get_color_frame()

                aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
                aligned_color_image = np.asanyarray(aligned_color_frame.get_data())


                while time_diff<2:
                    res = s.readline()
                    # print(res.decode("utf-8"))
                    res = res.decode("utf-8")
                    res = res.rstrip()
                    ser_arr.append(res)
                    current_time = time.time()
                    time_diff = current_time - start_time
                np.savetxt(path+"lidar/"+str(image_counter)+".csv", ser_arr, delimiter=", ", fmt="%s")
                cv2.imwrite(path +"images/"+ str(image_counter)+"_raw.jpg", color_image)
                np.savetxt(path+"depthmap/"+str(image_counter)+"_raw.csv", depth_image, delimiter=", ")
                cv2.imwrite(path +"depthmap/"+ str(image_counter)+"_raw.pgm", depth_image)
                np.savetxt(path+"depthmap/"+str(image_counter)+".csv", aligned_depth_image, delimiter=", ")
                cv2.imwrite(path +"depthmap/"+ str(image_counter)+".pgm", aligned_depth_image)
                if pc_flag:
                    ply = rs.save_to_ply(path +"pointcloud/"+ str(image_counter)+".ply")
                    ply.set_option(rs.save_to_ply.option_ply_binary, False)
                    ply.set_option(rs.save_to_ply.option_ply_normals, True)
                    ply.set_option(rs.save_to_ply.option_ply_mesh, True)
                    ply.process(colorized)
                
                break

        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.3), cv2.COLORMAP_JET)

        color_crop = color_image[centre[0]-100:centre[0]+100, centre[1]-100:centre[1]+100]
        depth_crop = aligned_depth_image[centre[0]-100:centre[0]+100, centre[1]-100:centre[1]+100]


        hist_depth = cv2.calcHist([depth_crop], [0], None, [np.max(depth_crop)], [1,np.max(depth_crop)])


        hist_depthfig = plt.figure(1)
        plt.plot(hist_depth)
        hist_depthfig.show()
        
        if model_flag:
            pred_im = predict_depth(model, color_image)
            pred_color = colorize(pred_im, cmap='inferno')

            pred_crop = (pred_im[centre[0]-100:centre[0]+100, centre[1]-100:centre[1]+100]*100)
            pred_crop = np.round(pred_crop)
            pred_max = int(np.max(pred_crop))
            # print(pred_max)
            hist_pred = cv2.calcHist([pred_crop], [0], None, [pred_max], [0,pred_max])

            hist_predfig = plt.figure(2)
            plt.plot(hist_pred)
            # plt.hist(pred_crop)
            hist_predfig.show()
            # print(pred_im[240, 320])
            print(mode(pred_crop, None))

            pred_color = cv2.rectangle(pred_color, start_p, end_p, color, thickness)
        
        
        # print(depth_image[240, 320])
        # print(mode(depth_crop[depth_crop>0], None))
        mode_list.append(mode(depth_crop[depth_crop>0], None)[0])
        print(mode_list)

        x_in = input()
        if x_in == 'q':
            loop_flag = False

    
        # Show images
        if showRS_flag:
            depth_colormap = cv2.rectangle(depth_colormap, start_p, end_p, color, thickness)
            images = np.hstack((color_rect, depth_colormap))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            if model_flag:
                cv2.imshow('ZoeDpeth', cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)

    finally:
        image_counter+=1
        continue

np.savetxt(path+"mode.csv", mode_list, delimiter=", ", fmt="%s")

pipeline.stop()