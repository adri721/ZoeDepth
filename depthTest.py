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


im_path = "C:/Users/ARajaraman/OneDrive - SharkNinja/Pictures/test.jpg"
filedir = "C:/Users/ARajaraman/OneDrive - SharkNinja/Pictures/"
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


color1 = [0, 255, 0]
centre = [320, 240]
start1 = (centre[0]-200, centre[1]-55)
end1 = (centre[0]+125, centre[1]-50)
thickness = 2

color2 = [255, 0, 0]
start2 = (centre[0]-200, centre[1]+80)
end2= (centre[0]+125, centre[1]+85)


pred_im = predict_depth(model, image)
pred_arr = np.asarray(pred_im)
pred_color = colorize(pred_im, cmap='inferno')
pred_color = cv2.rectangle(pred_color, start1, end1, color1, thickness)
pred_color = cv2.rectangle(pred_color, start2, end2, color2, thickness)
image = cv2.rectangle(image, start1, end1, color1, thickness)
image = cv2.rectangle(image, start2, end2, color2, thickness)
cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow('ZoeDpeth', cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
cv2.imwrite(filedir + "image.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite(filedir + "depth.jpg", cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
np.savetxt(filedir + "depth.csv", pred_arr, delimiter=", ")
cv2.waitKey(0)

pred_crop1 = (pred_im[centre[0]-200:centre[0]+125, centre[1]-55:centre[1]-50]*100)
pred_crop2 = (pred_im[centre[0]-200:centre[0]+125, centre[1]+80:centre[1]+85]*100)
pred_crop1 = np.round(pred_crop1)
pred_crop2 = np.round(pred_crop2)
pred_max1 = int(np.max(pred_crop1))
pred_max2 = int(np.max(pred_crop2))
# print(pred_max)
hist_pred1 = cv2.calcHist([pred_crop1], [0], None, [pred_max1], [0,pred_max1])
hist_pred2 = cv2.calcHist([pred_crop2], [0], None, [pred_max2], [0,pred_max2])

hist_predfig1 = plt.figure(1)
plt.plot(hist_pred1)
plt.title("Peak of red edge at:" + str(mode(pred_crop1, None)[0]) + "units")
hist_predfig2 = plt.figure(2)
plt.plot(hist_pred2)
plt.title("Peak of green edge at:" + str(mode(pred_crop2, None)[0]) + "units")
# plt.hist(pred_crop)
hist_predfig1.show()
hist_predfig2.show()
# print(pred_im[240, 320])
print(mode(pred_crop1, None))
print(mode(pred_crop2, None))

input()