import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import time
import cv2
import pgm_reader

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


##### loading Relative Depth Map model #####
conf = get_config("zoedepth", "infer")
DEVICE = 'cpu'

model = build_model(conf).to(DEVICE)
# model.eval()

def predict_depth(model, image):
    depth_im = model.infer_pil(image)
    x = ToTensor()(image).unsqueeze(0).to(DEVICE)
    # depth = model.infer(x)
    return depth_im



##### filepaths #####
# filedir = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/ZoeDepth/Intel_data/"
filedir = "Intel_data/"
# date_folder = "2024_4_17_17_2_3/"
date_folder = "2024_5_1_14_51_41/"
file_number = 0


##### CAMERA paramters #####
sensor_height = 1.512 #mm
sensor_width = 2.106

image_height = 480 #px
image_width = 640

f_cam = 1.88 #mm
pixel_to_mm = sensor_height/image_height




##### lidar-camera transform parameters ####
dx = 0
dy = 27.6
dz = 53.63

lidar_rgbCAM = 32.5 #b




##### READING the LiDAR data #####
subfolder = "lidar/"
lidar_path = filedir + date_folder + subfolder + str(file_number) + ".csv"

rows=[]

## reading raw data byte by byte
with open(lidar_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        try:
            f= float(row[0])
            rows.append(f)
        except:
            continue    

ii = 0

## the following are all lists and their defiition is given
## if the defition mentions a list, then the resultant is a list of lists

l_p=[]          # length of packet from the needle head to one byte before the check byte

ao=[]           # absolute angle offset
cfs=[]          # starting angle of currrent frame
cfe=[]          # end angle of current frame
rangA=[]        # list of ranging energies in the packet
rangB=[]        # list of ranging lengths in the packet
rangC=[]        # ranging lengths of the entire dataset
rangD=[]        # ranging energies of the entire dataset
anf=[]          # list of interpolated angles for the ranging lengths in the packet
angF=[]         # interpolated angles of the entire dataset to correspong for each ranging length 
angR=[]         # aforementioned data in radians 
angN=[]

mainList = []       # list of all the data in a packet (irrespective of which mode)    
Lsize = []          # length of the packet
sp=[]               # speed of LiDAR in that packet
ranList = []        # list of the data when LiDAR is in ranging mode
otherList = []      # list of the data when LiDAR is in not in the two modes

cnt = 0

for ii in range(len(rows)-200):
    if (rows[ii] == 170 and (rows[ii+2]-rows[ii+7])==8):
        lenP = int(rows[ii+1]*256 + rows[ii+2] + 2)
        Lsize.append(lenP)
        mainList.append(rows[ii:ii+lenP])
        cnt+=1
        
        speed = rows[ii+8]
        
        if(rows[ii+3]==16 and rows[ii+4]==97 and rows[ii+5]==173):
            ranList.append(rows[ii:ii+lenP])
            currentFrameStartAngle = rows[ii+11]*256 + rows[ii+12]
            cfs.append(abs((currentFrameStartAngle/100) -360))
            currentFrameEndAngle = rows[ii+13]*256 + rows[ii+14]
            cfe.append(abs((currentFrameEndAngle/100) -360))
            jj = 0
            ranging_a = []
            ranging_b = []
            while(jj<lenP - 18):
                # ranging_a = []
                # ranging_b = []
                ranging_a.append(rows[ii+jj+15])
                rangD.append(rows[ii+jj+15])
                ranging_b.append((rows[ii+jj+16]*256 + rows[ii+jj+17])/4)
                rangC.append((rows[ii+jj+16]*256 + rows[ii+jj+17])/4)
                
                jj+=3
            rangA.append(ranging_a)
            rangB.append(ranging_b)
            
            lenF = len(ranging_b)
            
            anf.append(np.linspace(abs((currentFrameStartAngle/100)), abs((currentFrameEndAngle/100)), lenF))
            for kk in anf[-1]:
                angF.append(kk)
                angR.append(kk*math.pi/180)
                
        else:
            otherList.append(rows[ii:ii+lenP])


angPlot = []
distPlot = []
scanCnt = 0
print(len(angF))
for ii in range(100, len(angF)):
    # if(angF[ii]>225 and angF[ii]<315):
    if(angF[ii]>225 and angF[ii]<315):
        angPlot.append(angR[ii])
        distPlot.append(rangC[ii])
        scanCnt= 1
    elif (scanCnt == 1):
        break


# plt.axes(projection = 'polar')
ax = plt.subplot(111, projection='polar')
#plotting the second scan data points in a polar plot
# fig1 = plt.figure(1)
ax.set_theta_direction(-1)
plt.plot(angPlot, distPlot)
plt.title('3i LiDAR distance plot')
plt.show()




##### Reading the RGB input image #####
subfolder = "images/"
im_path = filedir + date_folder + subfolder + str(file_number) + "_raw.jpg"

input_image = np.array(PIL.Image.open(im_path))
cv2.imshow('Input image', cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)




##### Generating the Depthmap #####
pred_depth = predict_depth(model, input_image)
pred_arr = np.asarray(pred_depth)
pred_color = colorize(pred_depth, cmap='inferno')

cv2.imshow('ZoeDpeth', cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)




##### Reading the DEPTHMAP data #####
subfolder = "depthmap/"
gt_path = filedir + date_folder + subfolder + str(file_number) + ".csv"

gt_depth = np.genfromtxt(gt_path, delimiter=',')
gt_colormap = cv2.applyColorMap(cv2.convertScaleAbs(gt_depth, alpha=0.35), cv2.COLORMAP_INFERNO)
cv2.imshow('Ground truth depth', gt_colormap)
cv2.waitKey(0)


color = [0, 255, 0]
centre = [320, 240]

thickness = 2

centre_angle = 277
start_angle = 279
end_angle = 275

lidar_dist = 0
ang_counter = 0
scanCnt = 0
for ii in range(100, len(angF)):
    if(angF[ii]>275 and angF[ii]<279):
        if rangC[ii] > 0:
            lidar_dist+= rangC[ii]
            print(rangC[ii])
            ang_counter+=1
            print(ang_counter)
            scanCnt = 1
    elif (scanCnt == 1):
        break

distance_lidar = lidar_dist/ang_counter
print(distance_lidar)


lidar_h_im = ((dy*f_cam)/(distance_lidar - dz))/pixel_to_mm
lidar_w_im = ((lidar_rgbCAM*f_cam)/(distance_lidar - dz))/pixel_to_mm

print(lidar_h_im)
print(lidar_w_im)

color = [0, 255, 0]
lidar_on_image = [int(lidar_w_im), int(lidar_h_im)]
start_l = (centre[0]+lidar_on_image[0]-2, centre[1]-lidar_on_image[1]-2)
end_l = (centre[0]+lidar_on_image[0]+2, centre[1]-lidar_on_image[1]+2)
thickness = 2


pred_rect = cv2.rectangle(pred_color, start_l, end_l, color, thickness)
cv2.imshow('ZoeDpeth', cv2.cvtColor(pred_rect, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)