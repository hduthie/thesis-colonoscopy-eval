import numpy as np
import matplotlib.pyplot as plt

def warp_kpts_with_depth_compare(gt_pose, est_pose, kpts0, depth_map, K, img0=None, img1=None, kpts1=None, frame_idx=0, step=30):
    K_inv = np.linalg.inv(K)

    # Sample depth (assumes depth in m)
    depths = depth_map[kpts0[:, 1].astype(int), kpts0[:, 0].astype(int)]
    depths *= 1000  # Convert to mm


    kpts0_h = np.hstack([kpts0, np.ones((kpts0.shape[0], 1))])
    normalized = (K_inv @ kpts0_h.T).T
    points_3d = normalized * depths[:, None]
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    def project(T):
        pts = (T @ points_3d_h.T).T[:, :3]
        proj = (K @ pts.T).T
        return proj[:, :2] / proj[:, 2:3]

    kpts_warped_gt = project(gt_pose)
    kpts_warped_est = project(est_pose)

    if img1 is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(img1)
        if kpts1 is not None:
            plt.scatter(kpts1[:, 0], kpts1[:, 1], s=10, c='red', label='Matched kpts1')
        plt.scatter(kpts_warped_gt[:, 0], kpts_warped_gt[:, 1], s=10, c='lime', label='GT Warped')
        plt.scatter(kpts_warped_est[:, 0], kpts_warped_est[:, 1], s=10, c='cyan', label='Est. Warped')
        plt.legend()
        plt.title(f"Frame {frame_idx} → {frame_idx+step}")
        plt.axis('off')
        plt.show()

    return kpts_warped_gt, kpts_warped_est
