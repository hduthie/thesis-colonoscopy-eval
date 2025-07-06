import pandas as pd
import numpy as np
from tqdm import tqdm

from data_utils import load_frame_pair
from pose_estimation import run_pose_estimation, compute_metrics
from visualization import plot_single_matcher_metrics, plot_combined_matcher_comparison, plot_trajectory_plotly
from trajectory_utils import compute_global_poses, extract_translation_vectors


def run_analysis(config):
    all_metrics = []
    all_poses = {}  # Store estimated poses per matcher

    for matcher_name in config['matchers']:
        print(f"Running matcher: {matcher_name}")
        matcher = config['matcher_loader']([matcher_name], device=config['device'], **config['ransac_kwargs'])

        metrics_list = []
        estimated_relative_poses = []

        i = 0
        while i < len(config['frame_paths']) - 1:
            success = False
            skip_used = 0

            for skip in range(1, config['max_skip'] + 1):
                if i + skip >= len(config['frame_paths']):
                    break

                img0, img1, depth0, depth1 = load_frame_pair(i, i + skip, matcher, config)
                result = matcher(img0, img1)
                num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']

                pose = run_pose_estimation(mkpts0, mkpts1, config['K'])
                if pose:
                    R, t = pose
                    T_est = np.eye(4)
                    T_est[:3, :3] = R
                    T_est[:3, 3] = t
                    estimated_relative_poses.append(T_est)
                    success = True
                    skip_used = skip
                    break

            if not success:
                i += 1
                continue

            # Compose GT pose from i to i+skip
            T_gt = np.eye(4)
            for k in range(i, i + skip_used):
                T_gt = config['gt_poses'][k] @ T_gt

            t_err, r_err, reproj_err = compute_metrics(
                R, t, T_gt, mkpts0, mkpts1, depth0, config['K'], img0, img1, i,
                visualize=config['visualize'] and (i % config['vis_every'] == 0)
            )

            metrics_list.append({
                'frame': i,
                'matcher': matcher_name,
                'num_inliers': num_inliers,
                'translation_error': t_err,
                'rotation_error': r_err,
                'skip': skip_used - 1,
                'reprojection_error': reproj_err
            })

            i += skip_used

        metrics_df = pd.DataFrame(metrics_list)
        all_metrics.append(metrics_df)
        all_poses[matcher_name] = estimated_relative_poses

        if config['visualize']:
            plot_single_matcher_metrics(metrics_df, matcher_name)

            poses = compute_global_poses(estimated_relative_poses)
            positions = extract_translation_vectors(poses)
            plot_trajectory_plotly(positions, title=f"Trajectory: {matcher_name}")

    if config['visualize']:
        plot_combined_matcher_comparison(all_metrics)

    return all_metrics, all_poses
