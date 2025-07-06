import os
import glob 
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import io

def undistort_images_and_depths(video_name, base_dir='data/C3VD'):
    """
    Undistorts frames and depth maps using fisheye intrinsics.
    Saves undistorted data to 'undistorted/' subfolders if not already present.
    """
    video_dir = os.path.join(base_dir, video_name)
    undistorted_frames_dir = os.path.join(video_dir, 'undistorted/frames')
    undistorted_depth_dir = os.path.join(video_dir, 'undistorted/depth')

    # Skip if already undistorted
    if os.path.exists(undistorted_frames_dir) and len(os.listdir(undistorted_frames_dir)) > 0 \
        and os.path.exists(undistorted_depth_dir) and len(os.listdir(undistorted_depth_dir)) > 0:
        print(f"[INFO] Undistorted data already exists for {video_name}. Skipping.")
        return

    print(f"[INFO] Undistorting video: {video_name}")
    os.makedirs(undistorted_frames_dir, exist_ok=True)
    os.makedirs(undistorted_depth_dir, exist_ok=True)

    # Load camera intrinsics and distortion coefficients
    K = np.load(os.path.join(base_dir, 'intrinsics.npy'))
    D = np.load(os.path.join(base_dir, 'dist_coeffs.npy'))

    # Load all frame and depth image paths
    frame_paths = sorted(glob.glob(os.path.join(video_dir, '*_color.png')))
    depth_paths = sorted(glob.glob(os.path.join(video_dir, '*_depth.tiff')))
    assert len(frame_paths) == len(depth_paths), "Frame and depth count mismatch!"

    # Undistort each frame and depth map
    for frame_path, depth_path in tqdm(zip(frame_paths, depth_paths), total=len(frame_paths)):
        filename = os.path.basename(frame_path)

        # Load and scale depth
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth_image = (depth_image / (2**16 - 1)) * 100  # normalize to meters

        # Load RGB frame
        img = cv2.imread(frame_path)
        h, w = depth_image.shape


        # Compute undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)

        # Undistort both depth and frame
        undistorted_depth = cv2.remap(depth_image, map1, map2, cv2.INTER_LINEAR)
        undistorted_frame = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # Save undistorted outputs
        cv2.imwrite(os.path.join(undistorted_depth_dir, filename.replace('.png', '.tiff')), undistorted_depth)
        cv2.imwrite(os.path.join(undistorted_frames_dir, filename), undistorted_frame)


def undistort_images_only(video_name, base_dir='data/C3VD'):
    """
    For the C3VD seq folders - real colonoscopy video without depth maps, only  images
    Undistorts RGB frames using fisheye intrinsics.
    Saves undistorted frames to 'undistorted/frames/' if not already present.
    """
    video_dir = os.path.join(base_dir, video_name)
    undistorted_frames_dir = os.path.join(video_dir, 'undistorted/frames')

    # Skip if already undistorted
    if os.path.exists(undistorted_frames_dir) and len(os.listdir(undistorted_frames_dir)) > 0:
        print(f"[INFO] Undistorted frames already exist for {video_name}. Skipping.")
        return

    print(f"[INFO] Undistorting RGB frames for video: {video_name}")
    os.makedirs(undistorted_frames_dir, exist_ok=True)

    # Load camera intrinsics and distortion coefficients
    K = np.load(os.path.join(base_dir, 'intrinsics.npy'))
    D = np.load(os.path.join(base_dir, 'dist_coeffs.npy'))

    # Load all frame image paths
    frame_paths = sorted(glob.glob(os.path.join(video_dir, '*_color.png')))

    for frame_path in tqdm(frame_paths, desc="Undistorting frames"):
        filename = os.path.basename(frame_path)

        img = cv2.imread(frame_path)
        if img is None:
            print(f"[WARNING] Failed to read image: {frame_path}")
            continue

        h, w = img.shape[:2]

        # Compute undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)

        # Undistort frame
        undistorted_frame = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # Save undistorted frame
        cv2.imwrite(os.path.join(undistorted_frames_dir, filename), undistorted_frame)


def load_images(video_name, image_dir='data/C3VD'):
    """
    Loads all undistorted image paths for a given video.
    """
    img_folder = os.path.join(image_dir, video_name, 'undistorted/frames')
    img_paths = sorted(glob.glob(os.path.join(img_folder, '*.png')))
    print(f"Found {len(img_paths)} images in {img_folder}")
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_folder}")
    return img_paths


def load_depths(video_name, depth_dir='data/C3VD'):
    """
    Loads all undistorted depth maps for a given video as float32 arrays.
    """
    depth_folder = os.path.join(depth_dir, video_name, 'undistorted/depth')
    depth_paths = sorted(glob.glob(os.path.join(depth_folder, '*.tiff')))
    print(f"Found {len(depth_paths)} depth maps in {depth_folder}")
    if not depth_paths:
        raise FileNotFoundError(f"No depth maps found in {depth_folder}")
    return [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) for p in depth_paths]



def load_gt_poses(video_name, pose_dir='data/C3VD', step=1, plot=False):
    """
    Loads ground-truth absolute camera poses (world-to-camera) and computes 
    relative poses (i → i+1) between consecutive frames.

    The input pose file contains camera-to-world poses, which are converted 
    to world-to-camera (i.e., global camera poses in the scene frame).
    
    Args:
        video_name (str): Sequence name
        pose_dir (str): Directory containing pose.txt
        step (int): Frame subsampling step
        plot (bool): Whether to plot the GT trajectory
    """
    pose_file = os.path.join(pose_dir, video_name, 'pose.txt')
    frame_dir = os.path.join(pose_dir, video_name, 'undistorted/frames')

    poses = []
    frame_paths = []

    # Load GT camera-to-world poses and convert to world-to-camera
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), step):
            values = list(map(float, lines[i].strip().split(',')))

            # Extract rotation and translation (camera-to-world)
            R_cam_to_world = np.array(values[:12]).reshape(3, 4)[:, :3]
            t_cam_to_world = np.array(values[12:15])

            # Convert to world-to-camera
            R_world_to_cam = R_cam_to_world.T
            t_world_to_cam = -R_world_to_cam @ t_cam_to_world

            # Construct 4x4 transformation matrix (T_world_to_cam)
            T = np.eye(4)
            T[:3, :3] = R_world_to_cam
            T[:3, 3] = t_world_to_cam

            frame_paths.append(Path(os.path.join(frame_dir, f"{i:04d}_color.png")))
            poses.append({'frame_path': frame_paths[-1], 'pose_matrix': T})

    # Compute relative poses T_rel = T1⁻¹ * T2 between consecutive frames
    rel_poses = []
    for i in range(len(poses) - 1):
        T1 = poses[i]['pose_matrix']
        T2 = poses[i + 1]['pose_matrix']
        rel = np.linalg.inv(T1) @ T2
        rel_poses.append(rel)

    # Plot trajectory if enabled
    if plot and len(poses) > 1:
        gt_xyz = np.array([T['pose_matrix'][:3, 3] for T in poses])

        images = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for angle in range(0, 360, 4):  # 90 frames at 4° increments
            ax.clear()
            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='GT Trajectory')
            ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], c='green', label='Start', s=40)
            ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], c='red', label='End', s=40)
            ax.view_init(elev=30, azim=angle)
            ax.set_title(f"GT Trajectory: {video_name}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc='upper right')

            # Render frame to image buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append(imageio.imread(buf))
            buf.close()

        gif_path = f"gt_traj_{video_name}.gif"
        imageio.mimsave(gif_path, images, fps=10)
        plt.close()
        print(f"[INFO] Saved GT trajectory rotation GIF to {gif_path}")


    return poses, rel_poses