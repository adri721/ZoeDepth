import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
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


# im_path = "C:/Users/ARajaraman/OneDrive - SharkNinja/Pictures/test.jpg"
filedir = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/ZoeDepth/Intel_data/2024_4_17_17_2_3/images/"
filename = "0"
im_path = filedir + filename +"_raw.jpg"
image = np.array(PIL.Image.open(im_path))

conf = get_config("zoedepth", "infer")
DEVICE = 'cpu'

model = build_model(conf).to(DEVICE)
# model.eval()

def predict_depth(model, image):
    depth_im = model.infer_pil(image)
    x = ToTensor()(image).unsqueeze(0).to(DEVICE)
    # depth = model.infer(x)
    return depth_im


pred_im = predict_depth(model, image)
pred_arr = np.asarray(pred_im)
pred_color = colorize(pred_im, cmap='inferno')

cv2.imshow('ZoeDpeth', cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
cv2.imwrite(filedir + filename + "_depth.jpg", cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
centre = [320, 240]
pred_crop = (pred_im[centre[0]-200:centre[0]+125, centre[1]-55:centre[1]-50]*100)
pred_max = int(np.max(pred_crop))# print(pred_max)
# hist_pred = cv2.calcHist([pred_crop], [0], None, [pred_max], [0,pred_max])

pred_crop = pred_crop*(273/144)
pred_max = int(np.max(pred_crop))# print(pred_max)
hist_pred = cv2.calcHist([pred_crop], [0], None, [pred_max], [0,pred_max])


hist_predfig = plt.figure(1)
plt.plot(hist_pred)
plt.title("Peak at:" + str(mode(pred_crop, None)[0]) + "units")
# plt.hist(pred_crop)
hist_predfig.show()

# print(pred_im[240, 320])
print(mode(pred_crop, None))

input()