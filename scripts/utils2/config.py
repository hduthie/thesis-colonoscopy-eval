# config.py
import torch
import numpy as np
from ../../libs/image_matching_models/matching import get_matcher, available_models

# === General Settings ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 512
undistort_first = True
visualize = True
max_skip = 5
vis_every = 10

# === Matcher Setup ===
chosen_matchers = ['gim-lg', 'disk-lg', 'xfeat']

def get_matcher(matcher_names, device, **kwargs):
    """Replace this with your actual matcher loading function."""
    raise NotImplementedError("Define how to load your matcher(s) here.")

# === Camera Intrinsics ===
# # parameters from undistortion code
fx = 767.3861511125845
fy = 767.5058656118406
cx = 679.054265997005
cy = 543.646891684636
k1 = -0.18867185058223412
k2 = -0.003927337093919806
k3 = 0.030524814153620117
k4 = -0.012756926010904904
# camera_params = (fx, fy, cx, cy, k1, k2, k3, k4)

# # Set camera intrinsic matrix
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# === Dataset Paths ===

raw_frame_paths = [...]  # e.g., list of original image file paths
raw_depth_paths = [...]  # corresponding depth maps
undistorted_frame_paths = [...]  # if using undistorted
undistorted_depths = [...]  # if using undistorted

# === Ground Truth Poses ===
gt_poses = [...]  # list of 4x4 np.arrays

# === RANSAC / Pose Estimation Parameters ===
ransac_kwargs = {
    'thresh': 0.5,
    'confidence': 0.999,
    'max_iters': 1000
}

# === Final Config Dictionary ===
config = {
    'device': device,
    'image_size': image_size,
    'undistort_first': undistort_first,
    'visualize': visualize,
    'matchers': chosen_matchers,
    'max_skip': max_skip,
    'vis_every': vis_every,
    'matcher_loader': dummy_get_matcher,
    'K': K,
    'raw_frame_paths': raw_frame_paths,
    'raw_depth_paths': raw_depth_paths,
    'undistorted_frame_paths': undistorted_frame_paths,
    'undistorted_depths': undistorted_depths,
    'gt_poses': gt_poses,
    'ransac_kwargs': ransac_kwargs
}