import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os


# ==============================
# 1. RELATIVE POSE ESTIMATION
# ==============================

def estimate_pose(kpts0, kpts1, K0, K1, thresh=0.5, conf=0.99999):
    """
    Estimates relative pose (R, t) between two views from matched keypoints.
    Uses the essential matrix and RANSAC to recover pose.

    This function is based on the work of Sierra Bonilla, UCL, with her permission.
    """
    if len(kpts0) < 5:
        return None

    # Normalize keypoints using intrinsics
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # Compute normalized RANSAC threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # Estimate essential matrix
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.USAC_MAGSAC)

    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # Recover pose from essential matrix
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0, _E)
            best_num_inliers = n

    return ret


def relative_pose_error(T_gt, R_est, t_est, ignore_gt_t_thr=0.0):
    """
    Computes rotation and translation angular errors between estimated and ground-truth poses.
    """
    t_gt = T_gt[:3, 3]
    t_mag = np.linalg.norm(t_est)
    t_gt_mag = np.linalg.norm(t_gt)

    # Translation angle error
    if t_mag < 1e-5 or t_gt_mag < 1e-5:
        t_err = 0 if t_gt_mag < ignore_gt_t_thr else 90.0
    else:
        cos_angle = np.abs(np.dot(t_est, t_gt)) / (t_mag * t_gt_mag)
        t_err = np.rad2deg(np.arccos(np.clip(cos_angle, 0.0, 1.0)))

    # Rotation angle error
    R_gt = T_gt[:3, :3]
    cos = (np.trace(np.dot(R_est.T, R_gt)) - 1) / 2
    R_err = np.rad2deg(np.abs(np.arccos(np.clip(cos, -1., 1.))))

    return t_err, R_err

def compute_reprojection_error(T12_est, mkpts0, mkpts1, depth0, K):
    """
    Projects 3D points from depth0 using estimated pose T12_est, compares with mkpts1.

    Returns:
        reprojection_error (np.ndarray): Reprojection error for each keypoint pair.
    """
    kpts0_h = np.hstack([mkpts0, np.ones((mkpts0.shape[0], 1))])
    K_inv = np.linalg.inv(K)
    normalized = (K_inv @ kpts0_h.T).T
    depths = depth0[mkpts0[:, 1].astype(int), mkpts0[:, 0].astype(int)]
    points_3d = normalized * depths[:, None]
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    warped = (K @ (T12_est @ points_3d_h.T)[:3]).T
    warped /= warped[:, 2:3]
    return np.linalg.norm(mkpts1 - warped[:, :2], axis=1)

# ==============================
# 2. SKIPPING FRAMES UTILITIES
# ==============================

class SkipLogger:
    def __init__(self):
        self.entries = []

    def log(self, frame_start, frame_end, skipped_frames, skip_used, success, reason=None):
        self.entries.append({
            'frame_start': frame_start,
            'frame_end': frame_end if success else None,
            'skipped_frames': skipped_frames,
            'skip_used': skip_used if success else None,
            'success': success,
            'reason': reason if not success else None
        })

    def save(self, out_path):
        df = pd.DataFrame(self.entries)
        df.to_csv(out_path, index=False)

# ==============================
# 3. TRAJECTORY UTILITIES
# ==============================
# def compute_global_poses(relative_poses, invert_rel=True):
#     """
#     Chains relative poses into global poses (camera-to-world).
#     If invert_rel=True, each relative pose is inverted before chaining (for world-to-camera T_rel).
#     """
#     T_global = [np.eye(4)]
#     for T_rel in relative_poses:
#         if invert_rel:
#             T_rel = np.linalg.inv(T_rel)
#         T_global.append(T_global[-1] @ T_rel)
#     return T_global

def compute_global_poses(relative_poses, multiply_order='T @ T_rel'):
    """
    Chains relative poses to global poses (camera-to-world).
    multiply_order: 'T @ T_rel' or 'T_rel @ T'
    """
    T_global = [np.eye(4)]
    for i, T_rel in enumerate(relative_poses):
        if multiply_order == 'T @ T_rel':
            # T_next = T_global[-1] @ T_rel
            # T_next = T_rel @ T_global[-1]
            T_next = T_global[-1] @ np.linalg.inv(T_rel)

        else:
            # T_next = T_rel @ T_global[-1]
            T_next = T_global[-1] @ np.linalg.inv(T_rel)

        T_global.append(T_next)
    return T_global


def extract_translation_vectors(T_list):
    """
    Extracts translation vectors from a list of 4x4 transformation matrices.
    """
    return np.array([T[:3, 3] for T in T_list])


def umeyama_alignment(X, Y):
    """
    Aligns two trajectories (in mm) using similarity transformation (scale, rotation, translation).
    """
    assert X.ndim == 2 and X.shape[1] == 3
    assert Y.ndim == 2 and Y.shape[1] == 3

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    U, S, Vt = svd(Xc.T @ Yc)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    s = np.sum(S) / np.trace(Xc.T @ Xc)
    t = mu_Y - s * R @ mu_X
    return s, R, t



# def evaluate_trajectory(rel_poses, gt_poses):
#     """
#     Computes Absolute Trajectory Error (ATE) in mm after aligning estimated trajectory to GT.
#     """
#     est_global = compute_global_poses(rel_poses)
#     gt_global = compute_global_poses(gt_poses)
#     est_xyz = extract_translation_vectors(est_global)
#     gt_xyz = extract_translation_vectors(gt_global)[:len(est_xyz)]
#     s, R, t = umeyama_alignment(est_xyz, gt_xyz)
#     aligned_xyz = (s * (R @ est_xyz.T)).T + t
#     ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)
#     return ate, est_xyz, aligned_xyz, gt_xyz







# def evaluate_trajectory(rel_poses, gt_pose_dict, pose_ids):
#     """
#     Computes ATE using only GT poses at the correct frame indices.
#     """
#     print(f"\n[DEBUG] Starting trajectory evaluation")
#     print(f"[DEBUG] Received {len(rel_poses)} relative estimated poses")
#     print(f"[DEBUG] Received {len(pose_ids)} GT pose_ids")

#     est_global = compute_global_poses(rel_poses)
#     print(f"[DEBUG] Global estimated poses: {len(est_global)} (includes identity)")

#     # Sanity check
#     if len(est_global) != len(pose_ids) + 1:
#         print(f"[WARNING] est_global has {len(est_global)} poses, but pose_ids has {len(pose_ids)} — slicing est_global[1:]")

#     # Discard identity pose (first one)
#     est_xyz = extract_translation_vectors(est_global[1:])
#     print(f"[DEBUG] est_xyz shape: {est_xyz.shape}")

#     # Extract matching GT global poses
#     # gt_global = [gt_pose_dict[pid] for pid in pose_ids]
#     gt_global = [np.linalg.inv(gt_pose_dict[pid]) for pid in pose_ids]
#     gt_xyz = extract_translation_vectors(gt_global)
#     print(f"[DEBUG] gt_xyz shape: {gt_xyz.shape}")

#     # Final shape check before alignment
#      # SAVE GIF BEFORE ALIGNMENT
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='GT')
#     ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label='Estimated (pre-Umeyama)')
#     ax.legend()
#     plt.title("Pre-Alignment Trajectory")

#     def update_raw(frame):
#         ax.view_init(elev=30, azim=frame)
#         return fig,

#     raw_gif_path = os.path.join("results", "raw_preumeyama.gif")  # or any path you prefer
#     ani = FuncAnimation(fig, update_raw, frames=range(0, 360, 4), interval=50)
#     ani.save(raw_gif_path, writer=PillowWriter(fps=20))
#     plt.close()
#     print(f"[INFO] Saved raw (pre-Umeyama) trajectory GIF to {raw_gif_path}")

#     print(f"[DEBUG] Running Umeyama alignment on {est_xyz.shape[0]} pose pairs...")

#     s, R, t = umeyama_alignment(est_xyz, gt_xyz)
#     aligned_xyz = (s * (R @ est_xyz.T)).T + t
#     ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)

#     print(f"[DEBUG] ATE stats → Mean: {ate.mean():.2f}, Median: {np.median(ate):.2f}, Max: {np.max(ate):.2f}\n")
#     return ate, est_xyz, aligned_xyz, gt_xyz

# # def evaluate_trajectory(rel_poses, gt_pose_dict, pose_ids):
# #     """
# #     Computes ATE using only GT poses at the correct frame indices.
# #     """
# #     est_global = compute_global_poses(rel_poses)
# #     est_xyz = extract_translation_vectors(est_global)

# #     # Only select GT poses for which we have valid rel_poses
# #     gt_global = [gt_pose_dict[pid] for pid in pose_ids]
# #     gt_xyz = extract_translation_vectors(gt_global)

# #     # Align and compute ATE
# #     s, R, t = umeyama_alignment(est_xyz, gt_xyz)
# #     aligned_xyz = (s * (R @ est_xyz.T)).T + t
# #     ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)
# #     return ate, est_xyz, aligned_xyz, gt_xyz




def evaluate_trajectory(rel_poses, gt_pose_dict, pose_ids, out_dir="results", multiply_order='T @ T_rel'):
    """
    Evaluates estimated trajectory vs GT using ATE, after Umeyama alignment.
    Automatically saves a pre-alignment visualization for debugging.
    """
    print(f"\n[INFO] Starting trajectory evaluation")
    print(f"[INFO] Relative poses: {len(rel_poses)}, Pose IDs: {len(pose_ids)}")

    # Compute estimated global trajectory
    est_global = compute_global_poses(rel_poses, multiply_order)
    if len(est_global) != len(pose_ids) + 1:
        print(f"[WARNING] est_global length mismatch: {len(est_global)} vs {len(pose_ids)} + 1")
    est_xyz = extract_translation_vectors(est_global[1:])
    print(f"[DEBUG] est_xyz.shape: {est_xyz.shape}")

    # Load GT poses, and try both directions to check
    print(f"[INFO] Checking GT poses at pose_ids[0]: {pose_ids[0]}")
    print(gt_pose_dict[pose_ids[0]])

    gt_raw = [gt_pose_dict[pid] for pid in pose_ids]
    gt_inv = [np.linalg.inv(gt_pose_dict[pid]) for pid in pose_ids]

    gt_raw_xyz = extract_translation_vectors(gt_raw)
    gt_inv_xyz = extract_translation_vectors(gt_inv)

    raw_diff = np.linalg.norm(est_xyz - gt_raw_xyz, axis=1).mean()
    inv_diff = np.linalg.norm(est_xyz - gt_inv_xyz, axis=1).mean()

    if raw_diff < inv_diff:
        print(f"[INFO] Using GT poses as camera-to-world (no inverse) [diff: {raw_diff:.2f} < {inv_diff:.2f}]")
        gt_xyz = gt_raw_xyz
    else:
        print(f"[INFO] Using GT poses as world-to-camera (inverted) [diff: {inv_diff:.2f} < {raw_diff:.2f}]")
        gt_xyz = gt_inv_xyz

    # # Pre-alignment visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='GT')
    # ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label='Estimated')
    # ax.legend()
    # plt.title("Trajectory (Pre-Alignment)")

    # def update_view(frame):
    #     ax.view_init(elev=30, azim=frame)
    #     return fig,

    # os.makedirs(out_dir, exist_ok=True)
    # gif_path = os.path.join(out_dir, "trajectory_pre_alignment.gif")
    # ani = FuncAnimation(fig, update_view, frames=range(0, 360, 4), interval=50)
    # ani.save(gif_path, writer=PillowWriter(fps=20))
    # plt.close()
    # print(f"[INFO] Saved pre-alignment trajectory GIF to {gif_path}")

    # Alignment
    print(f"[INFO] Running Umeyama alignment...")
    s, R, t = umeyama_alignment(est_xyz, gt_xyz)
    aligned_xyz = (s * (R @ est_xyz.T)).T + t
    ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)

    print(f"[RESULT] ATE mean: {ate.mean():.2f}, median: {np.median(ate):.2f}, max: {np.max(ate):.2f}")
    return ate, est_xyz, aligned_xyz, gt_xyz




# def evaluate_trajectory(rel_poses, gt_pose_dict, pose_ids):
#     """
#     Evaluates ATE after aligning estimated full poses using Umeyama.
#     """
#     print(f"[INFO] Starting trajectory evaluation...")
#     print(f"[INFO] Relative poses: {len(rel_poses)}, Pose IDs: {len(pose_ids)}")

#     # Build global trajectory from relative poses
#     est_global = compute_global_poses(rel_poses, 'true')
#     est_xyz = extract_translation_vectors(est_global[1:])

#     # Build GT trajectory
#     gt_global = [gt_pose_dict[pid] for pid in pose_ids]
#     gt_xyz = extract_translation_vectors(gt_global)

#     # Umeyama alignment on XYZs
#     print(f"[INFO] Running Umeyama alignment...")
#     s, R, t = umeyama_alignment(est_xyz, gt_xyz)

#     # Apply full-pose alignment
#     est_global_aligned = []
#     for T in est_global:
#         T_aligned = np.eye(4)
#         T_aligned[:3, :3] = s * R @ T[:3, :3]
#         T_aligned[:3, 3] = s * R @ T[:3, 3] + t
#         est_global_aligned.append(T_aligned)

#     aligned_xyz = extract_translation_vectors(est_global_aligned[1:])
#     ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)

#     print(f"[RESULT] ATE mean: {ate.mean():.2f}, median: {np.median(ate):.2f}, max: {np.max(ate):.2f}")
#     return ate, est_xyz, aligned_xyz, gt_xyz





# def evaluate_trajectory(rel_poses, gt_pose_dict, pose_ids, out_dir="results", multiply_order='T @ T_rel'):
#     """
#     Evaluates estimated trajectory vs GT using ATE, after Umeyama alignment.
#     """
#     # Compute estimated global trajectory
#     est_global = compute_global_poses(rel_poses, multiply_order)
#     est_xyz = extract_translation_vectors(est_global[1:])

#     # Use GT poses as camera-to-world
#     gt_xyz = extract_translation_vectors([gt_pose_dict[pid] for pid in pose_ids])

#     # Umeyama alignment
#     s, R, t = umeyama_alignment(est_xyz, gt_xyz)
#     aligned_xyz = (s * (R @ est_xyz.T)).T + t
#     ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)

#     print(f"[RESULT] ATE mean: {ate.mean():.2f}, median: {np.median(ate):.2f}, max: {np.max(ate):.2f}")
#     return ate, est_xyz, aligned_xyz, gt_xyz
