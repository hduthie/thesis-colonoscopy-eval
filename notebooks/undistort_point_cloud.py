import cv2
import numpy as np
import open3d as o3d
import os
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import copy

def depth_image_to_point_cloud(depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, voxel_size=None, batch_size=10, max_points=30000):
    # Read depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    
    # Convert depth image to float and scale
    depth_image = (depth_image.astype(np.float32)/(2**16-1))*100
    
    # Print depth image info
    print(f"Depth image shape: {depth_image.shape}, data type: {depth_image.dtype}")
    print(f"Depth value range: {np.min(depth_image)} - {np.max(depth_image)}")
    
    # Get depth image size
    height, width = depth_image.shape
    
    # Set camera intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Set distortion coefficients - for fisheye camera model
    D = np.array([k1, k2, k3, k4])
    
    try:
        # Undistort using fisheye camera model
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
        undistorted_depth = cv2.remap(depth_image, map1, map2, cv2.INTER_LINEAR)
        del depth_image, map1, map2
    except Exception as e:
        print(f"Undistortion failed, using original depth image: {e}")
        undistorted_depth = depth_image
        del depth_image
    
    # Create a mask of valid depth pixels
    valid_mask = undistorted_depth > 0
    valid_depth_count = np.sum(valid_mask)
    print(f"Number of valid depth pixels: {valid_depth_count}")
    
    if valid_depth_count == 0:
        print("Warning: No valid depth values found")
        return np.array([[0, 0, 0]])
    
    # Vectorized point cloud computation (instead of nested loops)
    v, u = np.indices((height, width))
    
    z = undistorted_depth[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    # Compute 3D coordinates
    x = (u_valid - cx) / fx * z
    y = (v_valid - cy) / fy * z
    
    point_cloud = np.column_stack((x, y, z))
    
    del undistorted_depth, valid_mask, v, u, z, u_valid, v_valid, x, y
    
    if voxel_size is not None and voxel_size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        point_cloud = np.asarray(pcd.points)
        del pcd
    
    print(f"Generated point cloud contains {len(point_cloud)} points")
    return point_cloud

def compute_range(point_cloud):
    """Compute diagonal length of bounding box of the point cloud."""
    if len(point_cloud) == 0:
        return 0.0
    min_xyz = np.min(point_cloud, axis=0)
    max_xyz = np.max(point_cloud, axis=0)
    return np.linalg.norm(max_xyz - min_xyz)

def load_poses(pose_path):
    """Load poses from a pose.txt file (column-major 4×4 matrices)."""
    print(f"Loading: {pose_path}")
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = list(map(float, line.strip().split(',')))
            if len(data) != 16:
                raise ValueError(f"Incorrect pose line length, expected 16 values, got {len(data)}: {line}")
            
            T = np.zeros((4, 4))
            for i in range(4):  # column
                for j in range(4):  # row
                    T[j, i] = data[i * 4 + j]
            
            if not np.allclose(T[3, :], [0, 0, 0, 1]):
                raise ValueError(f"Invalid transformation matrix, last row must be [0,0,0,1]:\n{T}")
            
            poses.append(T)
            
    print(f"Successfully loaded {len(poses)} pose matrices")
    return poses

def extract_rotation_translation_from_pose(pose):
    """Extract rotation matrix and translation vector from pose."""
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return rotation, translation

def apply_scale_to_point_cloud(pcd, scale_factor):
    """Apply scaling to point cloud centered at the origin."""
    pcd_scaled = copy.deepcopy(pcd)
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    pcd_scaled.transform(scale_matrix)
    return pcd_scaled

def get_rotation_matrix(rotation):
    """Create 4×4 transformation matrix from 3×3 rotation matrix."""
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rotation
    return rot_matrix

def get_translation_matrix(translation):
    """Create 4×4 transformation matrix from translation vector."""
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = translation
    return trans_matrix

def extract_scene_name(file_path, root_folder):
    """Extract scene name from file path."""
    rel_path = os.path.relpath(file_path, root_folder)
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return None

def find_pose_file(pose_root, scene_name):
    """Find pose.txt for a specific scene."""
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    if os.path.exists(pose_path):
        return pose_path
    return None

def process_single_file(file_path, output_folder, root_folder, camera_params, voxel_size, max_points, pose_matrix=None, apply_transform=False):
    try:
        print(file_path)
        fx, fy, cx, cy, k1, k2, k3, k4 = camera_params
        point_cloud = depth_image_to_point_cloud(
            file_path, fx, fy, cx, cy, k1, k2, k3, k4, 
            voxel_size=voxel_size, max_points=max_points
        )

        pc_range = compute_range(point_cloud)
        print(f"Point cloud range: {pc_range:.2f}")
        
        if len(point_cloud) <= 1:
            print(f"Warning: {os.path.basename(file_path)} generated empty/default point cloud, skipping")
            return False

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        if apply_transform and pose_matrix is not None:
            original_center = pcd.get_center()
            print(f"Original point cloud center: {original_center}")
            
            rotation, translation = extract_rotation_translation_from_pose(pose_matrix)
            
            print(f"Using pose:")
            print(f"Rotation matrix:\n{rotation}")
            print(f"Translation vector: {translation}")
            
            rotation_matrix = get_rotation_matrix(rotation)
            rotated_pcd = copy.deepcopy(pcd)
            rotated_pcd.transform(rotation_matrix)
            rotated_center = rotated_pcd.get_center()
            print(f"Rotated point cloud center: {rotated_center}")
            
            translation_matrix = get_translation_matrix(translation)
            transformed_pcd = copy.deepcopy(rotated_pcd)
            transformed_pcd.transform(translation_matrix)
            final_center = transformed_pcd.get_center()
            print(f"Final point cloud center: {final_center}")
            
            pcd = transformed_pcd
        
        root_dir = os.path.dirname(file_path)
        rel_path = os.path.relpath(root_dir, root_folder)
        target_folder = os.path.join(output_folder, rel_path)
        os.makedirs(target_folder, exist_ok=True)

        filename = os.path.basename(file_path)
        pc_file_name = filename[:-5] + '_pcd.ply'
        pc_file_path = os.path.join(target_folder, pc_file_name)
        o3d.io.write_point_cloud(pc_file_path, pcd)
        
        return True
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return False
    finally:
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Generate point cloud from depth images')
    parser.add_argument('--input', type=str, required=True, help='Root folder path of input depth images')
    parser.add_argument('--output', type=str, required=True, help='Root folder path to save output point clouds')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size for downsampling point cloud')
    parser.add_argument('--max_points', type=int, default=30000, help='Maximum number of points in point cloud')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for parallel processing')
    parser.add_argument('--pose_path', type=str, help='Path to pose file (single file)')
    parser.add_argument('--pose_root', type=str, help='Root folder containing pose files for each scene')
    parser.add_argument('--apply_transform', action='store_true', help='Apply rotation and translation transforms')
    args = parser.parse_args()
    
    fx = 767.3861511125845
    fy = 767.5058656118406
    cx = 679.054265997005
    cy = 543.646891684636
    k1 = -0.18867185058223412
    k2 = -0.003927337093919806
    k3 = 0.030524814153620117
    k4 = -0.012756926010904904
    camera_params = (fx, fy, cx, cy, k1, k2, k3, k4)
    
    root_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    voxel_size = args.voxel_size
    max_points = args.max_points
    
    depth_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_depth.tiff'):
                depth_files.append(os.path.join(root, file))
    
    total_files = len(depth_files)
    print(f"Found {total_files} depth image files to process")
    
    successful = 0
    
    if args.apply_transform:
        scene_poses = {}
        
        if args.pose_path and os.path.exists(args.pose_path):
            poses = load_poses(args.pose_path)
            print(f"Loaded {len(poses)} pose matrices from single pose file")
            for file_path in depth_files:
                scene_name = extract_scene_name(file_path, root_folder)
                if scene_name and scene_name not in scene_poses:
                    scene_poses[scene_name] = poses
        
        elif args.pose_root and os.path.exists(args.pose_root):
            print(f"Attempting to load pose files from root directory {args.pose_root}")
            scene_names = set()
            for file_path in depth_files:
                scene_name = extract_scene_name(file_path, root_folder)
                if scene_name:
                    scene_names.add(scene_name)
            
            for scene_name in scene_names:
                pose_file = find_pose_file(args.pose_root, scene_name)
                if pose_file:
                    print(f"Loading pose file for scene {scene_name}: {pose_file}")
                    scene_poses[scene_name] = load_poses(pose_file)
                else:
                    print(f"Warning: No pose file found for scene {scene_name}")
        
        else:
            print("Error: No valid pose file or pose root directory specified")
            return
        
        for file_path in depth_files:
            try:
                scene_name = extract_scene_name(file_path, root_folder)
                frame_idx = int(os.path.basename(file_path).split('_')[0])
                
                if scene_name in scene_poses:
                    poses = scene_poses[scene_name]
                    
                    if frame_idx < len(poses):
                        print(f"Processing file: {file_path} (Scene: {scene_name}, Frame: {frame_idx})")
                        result = process_single_file(
                            file_path, 
                            output_folder, 
                            root_folder,
                            camera_params,
                            voxel_size,
                            max_points,
                            pose_matrix=poses[frame_idx],
                            apply_transform=True
                        )
                        if result:
                            successful += 1
                    else:
                        print(f"Warning: Frame index {frame_idx} out of range for scene {scene_name} ({len(poses)} poses), skipping transform")
                        result = process_single_file(
                            file_path, 
                            output_folder, 
                            root_folder,
                            camera_params,
                            voxel_size,
                            max_points
                        )
                        if result:
                            successful += 1
                else:
                    print(f"Warning: No pose data found for scene {scene_name}, skipping transform")
                    result = process_single_file(
                        file_path, 
                        output_folder, 
                        root_folder,
                        camera_params,
                        voxel_size,
                        max_points
                    )
                    if result:
                        successful += 1
            except Exception as e:
                print(f"Error processing file {os.path.basename(file_path)}: {e}")
                continue
    else:
        process_file_fn = partial(
            process_single_file, 
            output_folder=output_folder, 
            root_folder=root_folder,
            camera_params=camera_params,
            voxel_size=voxel_size,
            max_points=max_points
        )
        
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(process_file_fn, depth_files))
            successful = sum(1 for r in results if r)
    
    print(f"Processing complete: {successful}/{total_files} files successfully processed")

if __name__ == '__main__':
    main()


#python your_script.py \
#   --input /path/to/depth_images \
#   --output /path/to/output_pcds \
#   --pose_path /path/to/pose.txt \
#   --apply_transform

# data/C3VD/cecum_t2_b/0415_depth.tiff
