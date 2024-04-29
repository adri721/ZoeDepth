import pyrealsense2 as rs
import numpy as np
import cv2

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


conf = get_config("zoedepth", "infer")
DEVICE = 'cpu'

model = build_model(conf).to(DEVICE)
# model.eval()

def predict_depth(model, image):
    depth_im = model.infer_pil(image)
    x = ToTensor()(image).unsqueeze(0).to(DEVICE)
    # depth = model.infer(x)
    return depth_im


def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


def get_mesh(model, image, keep_edges=False):
    # image.thumbnail((1024,1024))  # limit the size of the input image
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)

    # Create a trimesh mesh from the points
    # Each pixel is connected to its 4 neighbors
    # colors are the RGB values of the image

    verts = pts3d.reshape(-1, 3)
    image = np.array(image)
    if keep_edges:
        triangles = create_triangles(image.shape[0], image.shape[1])
    else:
        triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
    colors = image.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    return mesh



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

# Start streaming
pipeline.start(config)

loop_flag = True

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
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.rectangle(color_image, start_p, end_p, color, thickness)
        cv2.imshow('camera', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_INFERNO)

    pred_im = predict_depth(model, color_image)
    pred_color = colorize(pred_im, cmap='inferno')

    color_crop = color_image[centre[0]-100:centre[0]+100, centre[1]-100:centre[1]+100]
    depth_crop = depth_image[centre[0]-100:centre[0]+100, centre[1]-100:centre[1]+100]
    pred_crop = (pred_im[centre[0]-100:centre[0]+100, centre[1]-100:centre[1]+100]*100)
    pred_crop = np.round(pred_crop)

    pred_max = int(np.max(pred_crop))
    # print(pred_max)

    hist_depth = cv2.calcHist([depth_crop], [0], None, [np.max(depth_crop)], [1,np.max(depth_crop)])
    hist_pred = cv2.calcHist([pred_crop], [0], None, [pred_max], [0,pred_max])


    hist_depthfig = plt.figure(1)
    plt.plot(hist_depth)
    hist_depthfig.show()
    
    hist_predfig = plt.figure(2)
    plt.plot(hist_pred)
    # plt.hist(pred_crop)
    hist_predfig.show()
    # print(pred_im[240, 320])
    print(mode(pred_crop, None))
    # print(depth_image[240, 320])
    print(mode(depth_crop[depth_crop>0], None))

    input()

    color_image = cv2.rectangle(color_image, start_p, end_p, color, thickness)
    depth_colormap = cv2.rectangle(depth_colormap, start_p, end_p, color, thickness)
    pred_color = cv2.rectangle(pred_color, start_p, end_p, color, thickness)
    images = np.hstack((color_image, depth_colormap))

  
    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.imshow('ZoeDpeth', cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


finally:

    # Stop streaming
    pipeline.stop()