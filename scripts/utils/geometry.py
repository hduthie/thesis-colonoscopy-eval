import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    """
    Estimate relative pose using PnP (3D from kpts0, 2D from kpts1).
    """
    if len(kpts0) < 5:
        return None

    # try normalizing by max of both K0 and K1 instead
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.USAC_MAGSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0, _E)
            best_num_inliers = n

    return ret

def relative_pose_error(T_0to1, R_est, t_est, ignore_gt_t_thr=0.0):
    t_gt = T_0to1[:3,3]
    t_mag = np.linalg.norm(t_est)
    t_gt_mag = np.linalg.norm(t_gt)

    if t_mag < 1e-5 or t_gt_mag < 1e-5:
        if t_gt_mag < ignore_gt_t_thr:
            t_err = 0
        else:
            t_err = 90.0
    else:
        cos_angle = np.abs(np.dot(t_est, t_gt)) / (t_mag * t_gt_mag)
        cos_angle = np.clip(cos_angle, 0.0, 1.0)
        t_err = np.rad2deg(np.arccos(cos_angle))

    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R_est.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err