import numpy as np
import torch
from kiui.cam import orbit_camera
from torchvision.transforms.v2 import Pad


def pad_to_square(image):
    """Pads the image to make it square."""
    w, h = image.size  # Get the original width and height
    max_dim = max(w, h)
    padding = (
        (max_dim - w) // 2,  # Left
        (max_dim - h) // 2,  # Top
        (max_dim - w + 1) // 2,  # Right
        (max_dim - h + 1) // 2  # Bottom
    )
    return Pad(padding)(image)


def get_camera(
    num_frames, elevation=15, azimuth_start=0, azimuth_span=360, blender_coord=True):
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span + azimuth_start, angle_gap):

        pose = orbit_camera(-elevation, azimuth, radius=1) # kiui's elevation is negated, [4, 4]

        # opengl to blender, true
        if blender_coord:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]

        cameras.append(pose.flatten())

    out = torch.from_numpy(np.stack(cameras, axis=0))
    return out.float() # [num_frames, 16]