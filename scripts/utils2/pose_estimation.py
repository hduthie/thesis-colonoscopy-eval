import numpy as np
import cv2
from metrics import relative_pose_error
from visualization import warp_kpts_with_depth_compare

def run_pose_estimation(mkpts0, mkpts1, K, threshold=0.5):
    try:
        E, inliers = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, threshold=threshold, prob=0.999)
        if E is None or inliers is None:
            return None
        _, R, t, _ = cv2.recoverPose(E, mkpts0, mkpts1, K)
        return R, t.squeeze()
    except Exception as e:
        print(f"[ERROR] Pose estimation failed: {e}")
        return None

def compute_metrics(R, t, T_gt, mkpts0, mkpts1, depth0, K, img0, img1, frame_idx, visualize=False):
    T_est = np.eye(4)
    T_est[:3, :3] = R
    T_est[:3, 3] = t
    T_gt_inv = np.linalg.inv(T_gt)

    _, _, reprojection_error = warp_kpts_with_depth_compare(
        gt_pose=T_gt_inv,
        est_pose=T_est,
        kpts0=mkpts0,
        depth_map=depth0,
        K=K,
        img0=img0.permute(1, 2, 0).cpu().numpy(),
        img1=img1.permute(1, 2, 0).cpu().numpy(),
        kpts1=mkpts1,
        frame_idx=frame_idx,
        step=30,
        visualize=visualize
    )

    t_err, r_err = relative_pose_error(T_gt_inv, R, t, ignore_gt_t_thr=0.0)
    return t_err, r_err, reprojection_error
