import numpy as np
from sklearn.neighbors import NearestNeighbors

def combine_matchers(img0, img1, matchers, top_frac=0.33, deduplicate=False, dedup_thresh=2.0):
    """
    Combine multiple matchers' outputs into a single set of keypoint matches.

    Args:
        img0, img1: Input images (tensor or array depending on matcher)
        matchers: List of matcher instances (each callable as matcher(img0, img1))
        top_frac: Fraction of top matches to keep from each matcher (0.0–1.0)
        deduplicate: Whether to deduplicate overlapping matches across matchers
        dedup_thresh: Pixel distance threshold for deduplication

    Returns:
        mkpts0: np.ndarray of shape (N, 2)
        mkpts1: np.ndarray of shape (N, 2)
    """
    all_mkpts0, all_mkpts1 = [], []

    for matcher in matchers:
        result = matcher(img0, img1)
        mkpts0, mkpts1 = result['inlier_kpts0'], result['inlier_kpts1']
        print(f"[Ensemble] Matcher {matcher.__class__.__name__}: {len(mkpts0)} matches (before top_frac filter)")

        scores = result.get('match_scores', np.ones(len(mkpts0)))

        if top_frac < 1.0 and len(scores) > 0:
            top_k = int(len(scores) * top_frac)
            top_idx = np.argsort(-scores)[:top_k]
            mkpts0 = mkpts0[top_idx]
            mkpts1 = mkpts1[top_idx]

        all_mkpts0.append(mkpts0)
        all_mkpts1.append(mkpts1)

    mkpts0 = np.vstack(all_mkpts0)
    mkpts1 = np.vstack(all_mkpts1)

    if deduplicate and len(mkpts0) > 1:
        print(f"[Ensemble] Total combined matches before dedup: {mkpts0.shape[0]}")
        mkpts0, mkpts1 = deduplicate_matches(mkpts0, mkpts1, threshold=dedup_thresh)

    return mkpts0, mkpts1

# def deduplicate_matches(mkpts0, mkpts1, threshold=10.0):
#     matches = np.hstack([mkpts0, mkpts1])  # (N, 4) coordinates
#     nn = NearestNeighbors(radius=threshold).fit(matches)
#     idxs = nn.radius_neighbors(matches, return_distance=False)

#     for i, group in enumerate(idxs):
#         print(f"[DEBUG] group[{i}] = {group}")  # should print array of integer indices

#     # matches = np.hstack([mkpts0, mkpts1])  # shape (N, 4)
#     # nn = NearestNeighbors(radius=threshold).fit(matches)
#     # idxs = nn.radius_neighbors(matches)
#     keep = sorted(set(int(i[0]) for i in idxs if len(i) > 0))
#     print(f"[Ensemble] Deduplicated to {len(keep)} matches")

#     return mkpts0[keep], mkpts1[keep]

def deduplicate_matches(mkpts0, mkpts1, threshold=10.0):
    n = len(mkpts0)
    keep = set()

    # Find neighbors in both spaces
    nn0 = NearestNeighbors(radius=threshold).fit(mkpts0)
    nn1 = NearestNeighbors(radius=threshold).fit(mkpts1)

    neighbors0 = nn0.radius_neighbors(mkpts0, return_distance=False)
    neighbors1 = nn1.radius_neighbors(mkpts1, return_distance=False)

    used = set()

    for i in range(n):
        if i in used:
            continue

        group0 = set(neighbors0[i])
        group1 = set(neighbors1[i])
        common = group0 & group1

        # Pick one representative (e.g., the lowest index)
        keep.add(i)

        # Mark all duplicates so we don't pick them again
        used.update(common)

    keep = sorted(keep)
    print(f"[Ensemble] Deduplicated to {len(keep)} matches")
    return mkpts0[keep], mkpts1[keep]


def run_ensemble_ablation(
    img0, img1, depth0, K, T_gt, frame_idx,
    matchers, thresholds=[1, 3, 5, 8, 12, 15, 20],
    top_fracs=np.arange(0.1, 1.1, 0.1),
    visualize=False
):
    """
    Run ablation study over deduplication threshold and top_frac values.

    Args:
        img0, img1: Input images (torch.Tensor)
        depth0: np.ndarray depth map
        K: Intrinsic matrix
        T_gt: Ground truth 4x4 pose
        frame_idx: Frame index for logging
        matchers: List of matcher objects
        thresholds: List of deduplication radii (in pixels)
        top_fracs: List or array of top_frac values to retain
        visualize: Whether to enable visualization in metric computation

    Returns:
        results_df: pandas DataFrame of metrics for each config
    """
    results = []

    for thresh in thresholds:
        for frac in top_fracs:
            print(f"🔬 Testing threshold={thresh}, top_frac={frac:.1f}")

            # Run ensemble matcher
            mkpts0, mkpts1 = combine_matchers(
                img0, img1,
                matchers=matchers,
                top_frac=frac,
                deduplicate=True,
                dedup_thresh=thresh,
            )

            if len(mkpts0) < 8:
                print("⚠️  Not enough matches, skipping.")
                continue

            # Estimate pose from matches
            try:
                R, t = estimate_pose(mkpts0, mkpts1, K)
            except Exception as e:
                print(f"❌ Pose estimation failed: {e}")
                continue

            try:
                t_err, r_err, reproj_err = compute_metrics(
                    R, t, T_gt, mkpts0, mkpts1, depth0, K, img0, img1, frame_idx,
                    visualize=visualize
                )
            except Exception as e:
                print(f"❌ Metric computation failed: {e}")
                continue

            results.append({
                "frame": frame_idx,
                "threshold": thresh,
                "top_frac": round(frac, 2),
                "rot_err": r_err,
                "trans_err": t_err,
                "reproj_err": reproj_err,
                "num_matches": len(mkpts0),
            })

    results_df = pd.DataFrame(results)
    return results_df
