import numpy as np

def load_frame_pair(index0, index1, matcher, config):
    """
    Load image pair and depth maps for matching.
    Respects whether to use undistorted or raw frames.
    """
    if config['undistort_first']:
        img0 = matcher.load_image(config['undistorted_frame_paths'][index0])
        img1 = matcher.load_image(config['undistorted_frame_paths'][index1])
        depth0 = config['undistorted_depths'][index0]
        depth1 = config['undistorted_depths'][index1]
    else:
        img0 = matcher.load_image(config['raw_frame_paths'][index0], resize=config['image_size'])
        img1 = matcher.load_image(config['raw_frame_paths'][index1], resize=config['image_size'])
        depth0 = config['raw_depth_paths'][index0]
        depth1 = config['raw_depth_paths'][index1]

    return img0, img1, depth0, depth1
