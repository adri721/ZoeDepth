import numpy as np
import trimesh
from zoedepth.utils.geometry import depth_to_points, create_triangles
from functools import partial
import tempfile
import PIL
from zoedepth.utils.misc import get_image_from_url, colorize
import torch
import torch.nn.functional as F

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from torchvision.transforms import ToTensor
from PIL import Image
import cv2
import time

conf = get_config("zoedepth", "infer")
DEVICE = 'cpu'

model = build_model(conf).to(DEVICE)
# model.eval()


def predict_depth(model, img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h, w = image.shape[:2]
    depth = model.infer_pil(image)
    depth = colorize(depth)
    # depth = depth.cpu().numpy().astype(np.uint8)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    
    return depth

webcam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
   start_time = time.time()
   ret, frame = webcam.read()

   heat = predict_depth(model, frame)
#    heat = frame

   end_time = time.time()
   secs = end_time - start_time
   fps = str(1/secs)
   print(fps)
   cv2.putText(heat, fps, (7, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)
   cv2.imshow('heatmap', heat)

   if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
webcam.release() 
cv2.destroyAllWindows() 