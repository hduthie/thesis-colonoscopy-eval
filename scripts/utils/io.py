import os
import glob 
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# def check_undistorted_exists(video_name, base_dir='data/C3VD'):
#     frames_dir = os.path.join(base_dir, video_name, 'undistorted/frames')
#     depth_dir = os.path.join(base_dir, video_name, 'undistorted/depth')
#     return os.path.isdir(frames_dir) and os.listdir(frames_dir) and os.path.isdir(depth_dir) and os.listdir(depth_dir)

def undistort_if_needed(video_name, base_dir='data/C3VD'):
    video_dir = os.path.join(base_dir, video_name)
    undistorted_frames_dir = os.path.join(video_dir, 'undistorted/frames')
    undistorted_depth_dir = os.path.join(video_dir, 'undistorted/depth')
 

    # Check if already done
    if os.path.exists(undistorted_frames_dir) and len(os.listdir(undistorted_frames_dir)) > 0 \
        and os.path.exists(undistorted_depth_dir) and len(os.listdir(undistorted_depth_dir)) > 0:
        print(f"[INFO] Undistorted data already exists for {video_name}. Skipping.")
        return

    print(f"[INFO] Undistorting video: {video_name}")

    os.makedirs(undistorted_frames_dir, exist_ok=True)
    os.makedirs(undistorted_depth_dir, exist_ok=True)


    # Load intrinsics and distortion
    K = np.load(os.path.join(base_dir, 'intrinsics.npy'))
    D = np.load(os.path.join(base_dir, 'dist_coeffs.npy'))

    # Get frame/depth file lists
    print("checking video dir", video_dir)
    frame_paths = sorted(glob.glob(os.path.join(video_dir, '*_color.png')))
    print("checking frame path", frame_paths)
    depth_paths = sorted(glob.glob(os.path.join(video_dir, '*_depth.tiff')))
    print("checking depth path", depth_paths)
    assert len(frame_paths) == len(depth_paths), "Frame and depth count mismatch!"

    for frame_path, depth_path in tqdm(zip(frame_paths, depth_paths), total=len(frame_paths)):
        filename = os.path.basename(frame_path)
        print("checking depth path", depth_path)    
        # Read and scale depth
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth_image = (depth_image / (2**16 - 1)) * 100

        # Read image
        img = cv2.imread(frame_path)
        h, w = depth_image.shape
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
        undistorted_depth = cv2.remap(depth_image, map1, map2, cv2.INTER_LINEAR)
        undistorted_frame = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # Save
        cv2.imwrite(os.path.join(undistorted_depth_dir, filename.replace('.png', '.tiff')), undistorted_depth)
        cv2.imwrite(os.path.join(undistorted_frames_dir, filename), undistorted_frame)


def load_images(video_name, image_dir='data/C3VD'):
    img_folder = os.path.join(image_dir, video_name, 'undistorted/frames')
    img_paths = sorted(glob.glob(os.path.join(img_folder, '*.png')))
    print(f"Found {len(img_paths)} images in {img_folder}")
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_folder}")
    return img_paths

def load_depths(video_name, depth_dir='data/C3VD'):
    depth_folder = os.path.join(depth_dir, video_name, 'undistorted/depth')
    depth_paths = sorted(glob.glob(os.path.join(depth_folder, '*.tiff')))
    print(f"Found {len(depth_paths)} depth maps in {depth_folder}")
    if not depth_paths:
        raise FileNotFoundError(f"No depth maps found in {depth_folder}")
    return [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) for p in depth_paths]

# def load_gt_poses(video_name, pose_dir='data/C3VD'):
#     pose_file = os.path.join(pose_dir, video_name, 'pose.txt')
#     poses = []
#     with open(pose_file, 'r') as f:
#         for line in f:
#             vals = list(map(float, line.strip().split(',')))
#             mat = np.array(vals).reshape(4, 4)
#             poses.append(mat)

#     gt_abs = [{'frame': i, 'pose_matrix': p} for i, p in enumerate(poses)]
#     gt_rel = [np.linalg.inv(gt_abs[i]['pose_matrix']) @ gt_abs[i+1]['pose_matrix'] for i in range(len(gt_abs)-1)]
#     gt_rel = [{'frame': i, 'pose_matrix': p} for i, p in enumerate(gt_rel)]
#     print(f"Loaded {len(gt_abs)} absolute poses and {len(gt_rel)} relative poses from {pose_file}")
#     if not gt_abs:
#         raise FileNotFoundError(f"No ground truth poses found in {pose_file}")
#     return gt_abs, gt_rel

def load_gt_poses(video_name, pose_dir='data/C3VD', step=1):
    pose_file = os.path.join(pose_dir, video_name, 'pose.txt')
    frame_dir = os.path.join(pose_dir, video_name, 'undistorted/frames')
    poses = []
    frame_paths = []

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), step):
            values = list(map(float, lines[i].strip().split(',')))

            # Camera-to-world
            R_cam_to_world = np.array(values[:12]).reshape(3, 4)[:, :3]
            t_cam_to_world = np.array(values[12:15])

            # Convert to world-to-camera
            R_world_to_cam = R_cam_to_world.T
            t_world_to_cam = -R_world_to_cam @ t_cam_to_world

            T = np.eye(4)
            T[:3, :3] = R_world_to_cam
            T[:3, 3] = t_world_to_cam

            frame_paths.append(Path(os.path.join(frame_dir, f"{i:04d}_color.png")))
            poses.append({'frame_path': frame_paths[-1], 'pose_matrix': T})

    # Compute relative poses T12 between consecutive frames
    rel_poses = []
    for i in range(len(poses) - 1):
        T1 = poses[i]['pose_matrix']
        T2 = poses[i + 1]['pose_matrix']
        rel = np.linalg.inv(T1) @ T2
        rel_poses.append(rel)

    return poses, rel_poses

