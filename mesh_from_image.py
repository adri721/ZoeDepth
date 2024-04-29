import numpy as np
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

conf = get_config("zoedepth", "infer")
DEVICE = 'cpu'

model = build_model(conf).to(DEVICE)
# model.eval()

# im_path = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/MiDaS/7c_88_99_0a_bf_b2/bfb2_20170804090133_92507_0.00_0.00_0.00_s.png"
im_path = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/calib_3_26/set1/bfb2_20170804090054_53100_0.00_0.00_0.00_s.png"
image = PIL.Image.open(im_path)
glb_path = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/ZoeDepth/"

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


def predict_depth(model, image):
    depth_im = model.infer_pil(image)
    x = ToTensor()(image).unsqueeze(0).to(DEVICE)
    depth = model.infer(x)
    return depth_im

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

    # # Save as glb
    # glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    # glb_path = glb_file.name
    # mesh.export(glb_path)
    # return glb_path

    
mesh = get_mesh(model, image)
mesh.export(glb_path+"mesh3.glb")
mesh.show()