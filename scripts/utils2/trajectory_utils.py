import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd

def compute_global_poses(relative_poses): #not used in matching_analysis.py
    """
    Composes global poses from a list of relative poses (T_i_to_i+1).
    Returns a list of global camera-to-world transforms.
    """
    T_global = [np.eye(4)]
    for T_rel in relative_poses:
        T_next = T_rel @ T_global[-1]
        T_global.append(T_next)
    return T_global

def extract_translation_vectors(T_list):
    """
    Extracts 3D translation vectors (camera positions) from a list of 4x4 poses.
    """
    return np.array([T[:3, 3] for T in T_list])

def umeyama_alignment(X, Y):  # X: est Nx3, Y: gt Nx3
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    U, S, Vt = svd(Xc.T @ Yc)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # Reflection case
        Vt[-1, :] *= -1
        R = U @ Vt
    s = np.trace(S) / np.trace(Xc.T @ Xc)
    t = mu_Y - s * R @ mu_X
    return s, R, t

def evaluate_trajectory(rel_poses, gt_poses):
    est_global = compute_global_poses(rel_poses)
    gt_global = compute_global_poses(gt_poses)
    est_xyz = extract_translation_vectors(est_global)
    gt_xyz = extract_translation_vectors(gt_global)[:len(est_xyz)]
    s, R, t = umeyama_alignment(est_xyz, gt_xyz)
    aligned_xyz = (s * (R @ est_xyz.T)).T + t
    ate = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)
    return ate, est_xyz, aligned_xyz, gt_xyz

