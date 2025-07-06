import argparse
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

from libs.image_matching_models.matching import get_matcher
from src.io import undistort_images_and_depths, undistort_images_only, load_images, load_depths, load_gt_poses
from src.ensemble import combine_matchers
from src.pose_utils import SkipLogger, estimate_pose, relative_pose_error, compute_reprojection_error, evaluate_trajectory


def run_analysis(video_name, matcher_name, out_dir, start=0, end=None, step=1, top_frac=None, dedup_thresh=None):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Load images and depths
    image_only_videos = {'seq1', 'seq2', 'seq3', 'seq4'}
    if video_name in image_only_videos:
        undistort_images_only(video_name)
    else:
        undistort_images_and_depths(video_name)
        depth_paths = load_depths(video_name)
    img_paths = load_images(video_name)

    # Load GT poses and intrinsics
    gt_abs_poses, gt_relative_poses = load_gt_poses(video_name, step=30)
    K = np.load(f"data/C3VD/intrinsics.npy")

    # Setup matchers
    if matcher_name == "ensemble":
        matcher_names = ["xfeat", "gim-lg", "disk-lg"]
        matcher_suffix = f"ensemble_topfrac{top_frac}_thresh{dedup_thresh}"
        matchers = [get_matcher(name, device=device) for name in matcher_names]
        is_ensemble = True
        print(f'TOP FRAC: {top_frac}, DEDUP THRESH: {dedup_thresh}')
    else:
        matcher_names = [matcher_name]
        matcher_suffix = matcher_name
        matchers = [get_matcher(matcher_name, device=device)]
        is_ensemble = False

    # Setup output directories
    video_out_dir = os.path.join(out_dir, video_name, matcher_suffix)
    relpose_dir = os.path.join(video_out_dir, 'poses')
    match_dir = os.path.join(video_out_dir, 'matches')
    gt_pose_dir = os.path.join(out_dir, video_name, 'gt_poses')
    os.makedirs(relpose_dir, exist_ok=True)
    os.makedirs(gt_pose_dir, exist_ok=True)
    os.makedirs(match_dir, exist_ok=True)

    if end is None:
        end = len(img_paths) - 1

    metrics = []
    pose_ids = [] # True pose IDs after skipping
    skip_logger = SkipLogger()
    max_skip = 5

    # Main processing loop
    i = start
    while i < min(end, len(img_paths) - 1):
        print(f"Processing frame {i} / {len(img_paths)}")
        success = False
        actual_skip = 1

        # Attempt to estimate pose between frame i and i + skip 
        for skip in range(1, max_skip + 1):
            if i + skip >= len(img_paths):
                break

            print(f"Attempting skip {skip} from frame {i} → {i + skip}")
            img0 = matchers[0].load_image(img_paths[i])
            img1 = matchers[0].load_image(img_paths[i + skip])
            depth0 = depth_paths[i] if 'depth_paths' in locals() else None

            # If ensemble matcher, combine matched keypoint results
            if is_ensemble:
                mkpts0, mkpts1 = combine_matchers(img0, img1, matchers, top_frac=top_frac, deduplicate=True, dedup_thresh=dedup_thresh)
                result = {
                    'inlier_kpts0': mkpts0,
                    'inlier_kpts1': mkpts1,
                    'num_inliers': len(mkpts0),
                }
            else:
                result = matchers[0](img0, img1)
                mkpts0, mkpts1 = result['inlier_kpts0'], result['inlier_kpts1']

            print(f"Found {len(mkpts0)} matches")
            if len(mkpts0) < 6:
                print(f"Insufficient matches (<6) for frame pair {i}->{i+skip}")
                continue

            # Estimate pose
            try:
                pose_result = estimate_pose(mkpts0, mkpts1, K, K, thresh=0.5)
                if pose_result is None:
                    raise ValueError("estimate_pose returned None")
                R, t, inliers, E = pose_result
                print(f"[SUCCESS] Pose estimated for frame pair {i}->{i+skip}")
                success = True
                actual_skip = skip
                break
            except Exception as e:
                print(f"[ERROR] Pose estimation failed at frame pair {i}->{i+skip}: {e}")

        #
        skipped = list(range(i + 1, i + actual_skip)) if success else list(range(i + 1, min(i + max_skip + 1, len(img_paths))))
        skip_logger.log(i, i + actual_skip if success else None, skipped, actual_skip if success else None, success, reason=None if success else "Pose estimation failed")

        if not success:
            print(f"[SKIP] Skipping frame {i} due to persistent failures")
            i += 1
            continue

        # Compose estimated transformation homogeneous matrix
        T12_est = np.eye(4)
        T12_est[:3, :3] = R
        T12_est[:3, 3] = t

        # Compute ground truth relative pose
        T12_gt = np.eye(4)
        for k in range(i, i + actual_skip):
            if k < len(gt_relative_poses):
                T12_gt = gt_relative_poses[k] @ T12_gt

        # Compute relative pose errors: rotation and translation
        t_err, r_err = relative_pose_error(np.linalg.inv(T12_gt), R, t)
        print(f"[{matcher_name}] Frame {i} → {i + actual_skip}: Rot err = {r_err:.2f}°, Trans err = {t_err:.2f}°")

        # Reprojection error
        if depth0 is None or not isinstance(depth0, np.ndarray):
            reproj_error = np.array([])
            mean_reproj_error = 0.0
        else:
            reproj_error = compute_reprojection_error(T12_est, mkpts0, mkpts1, depth0, K)
            mean_reproj_error = np.mean(reproj_error)

        metrics.append({
            'video': video_name,
            'matcher': matcher_suffix,
            'frame_start': i,
            'frame_end': i + actual_skip,
            'rotation_error_deg': r_err,
            'translation_error_deg': t_err,
            'num_matches': len(mkpts0),
            'num_inliers': result['num_inliers'],
            'mean_reprojection_error_px': mean_reproj_error,
        })

        # Save results
        data_step = 30
        id0, id1 = i * data_step, (i + actual_skip) * data_step
        pose_ids.append(id1)

        np.save(os.path.join(relpose_dir, f"T_{id0:05d}_{id1:05d}.npy"), T12_est)
        np.save(os.path.join(gt_pose_dir, f"T_gt_{id0:05d}_{id1:05d}.npy"), T12_gt)

        torch.save({
            'mkpts0': torch.from_numpy(mkpts0),
            'mkpts1': torch.from_numpy(mkpts1),
            'inliers': torch.from_numpy(inliers) if inliers is not None else None,
            'reprojection_errors': torch.from_numpy(reproj_error),
            'img0_path': Path(img_paths[i]),
            'img1_path': Path(img_paths[i + actual_skip]),
        }, os.path.join(match_dir, f"matches_{id0:05d}_{id1:05d}.torch"))

        i += actual_skip


    # ------------------------------
    # Final Evaluation Block 
    # ------------------------------

    print("\n========== Final Evaluation ==========")

    # Save pose IDs -> accounting for skipped frames
    pose_ids_path = os.path.join(video_out_dir, "frame_ids.npy")
    np.save(pose_ids_path, np.array(pose_ids))
    print(f"[INFO] Saved pose_ids to {pose_ids_path}")

    # Load estimated relative poses
    relpose_files = sorted(os.listdir(relpose_dir))
    rel_poses = [np.load(os.path.join(relpose_dir, f)) for f in relpose_files]
    print(f"[INFO] Loaded {len(rel_poses)} estimated relative poses")
    # for i, T in enumerate(rel_poses):
    #     print(f"[DEBUG] T_rel[{i}]:\n{T}\n")

    # Build GT pose dictionary: {frame_id: pose_matrix}
    gt_pose_dict = {int(i * 30): gt_abs_poses[i]['pose_matrix'] for i in range(len(gt_abs_poses))}
    print(f"[INFO] GT pose dict constructed with {len(gt_pose_dict)} entries")

    # Load and sanitize pose IDs
    pose_ids = np.load(pose_ids_path).tolist()
    pose_ids = [int(i) for i in pose_ids]
    print(f"[INFO] Loaded {len(pose_ids)} pose IDs")
    # print(f"[DEBUG] First few GT dict keys: {list(gt_pose_dict.keys())[:5]}")
    # print(f"[DEBUG] First few pose_ids: {pose_ids[:5]}")

    # Filter for valid (GT-aligned) pose pairs
    valid_pairs = [(pose, pid) for pose, pid in zip(rel_poses, pose_ids) if pid in gt_pose_dict]
    print(f"[INFO] Found {len(valid_pairs)} valid (estimated, GT) pose pairs after filtering")

    if not valid_pairs:
        raise ValueError("[ERROR] No valid pose pairs found for evaluation")

    # Unpack filtered pairs
    rel_poses, pose_ids = zip(*valid_pairs)
    rel_poses = list(rel_poses)
    pose_ids = list(pose_ids)
    # for i, T in enumerate(rel_poses[:3]):
    #     print(f"[DEBUG] T_rel[{i}]:\n{T}\n")


    print(f"[INFO] Evaluating ATE on {len(rel_poses)} temporally aligned poses")
    

    # Run evaluation
    ate, est_xyz, aligned_xyz, gt_xyz = evaluate_trajectory(rel_poses, gt_pose_dict, pose_ids)
    # print(f"[DEBUG] Frame-wise pose alignment:")
    # for i, pid in enumerate(pose_ids):
    #     print(f"  Pose {i:02d}: Frame {pid} | GT XYZ: {gt_xyz[i]} | Est XYZ: {est_xyz[i]}")

    # Save ATE values
    ate_path = os.path.join(video_out_dir, f"ate_{matcher_name}.npy")
    np.save(ate_path, ate)
    print(f"[INFO] Saved ATE to {ate_path}")
    print(f"[RESULT] ATE Mean for {matcher_name}: {ate.mean():.2f} mm")

    # # Create rotating trajectory animation
    if top_frac is not None and dedup_thresh is not None:
        if top_frac == 3 and dedup_thresh == 0.3:

            print("[INFO] Generating rotating trajectory GIF...")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot once (so limits are fixed)
            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='GT')
            ax.plot(aligned_xyz[:, 0], aligned_xyz[:, 1], aligned_xyz[:, 2], label='Estimated')
            ax.legend()
            plt.title(f"Trajectory: {matcher_name}")

            def update(frame):
                ax.view_init(elev=30, azim=frame)
                return fig,

            gif_path = os.path.join(video_out_dir, f"trajectory_{matcher_name}.gif")
            ani = FuncAnimation(fig, update, frames=range(0, 360, 4), interval=50)
            ani.save(gif_path, writer=PillowWriter(fps=8))
            plt.close()
            print(f"[INFO] Saved rotating trajectory GIF to {gif_path}")
    else:

        print("[INFO] Generating rotating trajectory GIF...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot once (so limits are fixed)
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='GT')
        ax.plot(aligned_xyz[:, 0], aligned_xyz[:, 1], aligned_xyz[:, 2], label='Estimated')
        ax.legend()
        plt.title(f"Trajectory: {matcher_name}")

        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return fig,

        gif_path = os.path.join(video_out_dir, f"trajectory_{matcher_name}.gif")
        ani = FuncAnimation(fig, update, frames=range(0, 360, 4), interval=50)
        ani.save(gif_path, writer=PillowWriter(fps=8))
        plt.close()
        print(f"[INFO] Saved rotating trajectory GIF to {gif_path}")

    # Save metrics
    metrics_path = os.path.join(video_out_dir, f"metrics_{matcher_name}.csv")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    print(f"[INFO] Saved metrics to {metrics_path}")

    # Save skip log
    skip_log_path = os.path.join(video_out_dir, f"skip_log_{matcher_name}.csv")
    skip_logger.save(skip_log_path)
    print(f"[INFO] Saved skip log to {skip_log_path}")

    print(f"[SUCCESS] All results saved to: {video_out_dir}")
    print("======================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', required=True, help='Name of the video')
    parser.add_argument('--matcher', required=True, help='Name of the matcher or "ensemble"')
    parser.add_argument('--out_dir', default='results', help='Output directory')
    parser.add_argument('--step', type=int, default=1, help='Step between frames')
    parser.add_argument('--top_frac', type=float, default=None, help='Top fraction of matches to keep (only for ensemble)')
    parser.add_argument('--dedup_thresh', type=int, default=None, help='Deduplication threshold (only for ensemble)')

    args = parser.parse_args()

    run_analysis(
        args.video_name, 
        args.matcher, 
        args.out_dir, 
        step=args.step,
        start=0,
        top_frac=args.top_frac,
        dedup_thresh=args.dedup_thresh
    )

