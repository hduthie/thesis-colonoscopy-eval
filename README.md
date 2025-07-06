

src/
├── __init__.py
├── io.py                  # data loading: images, depths, poses, intrinsics - undistortion
├── ensemble.py            # ensemble matching: match confidence scoring, selection and deduplication, ablation
├── pose_utils.py          # estimate_pose(), relative_pose_error(), pose chaining, umeyama alignment, ATE
├── metrics.py             # saving results (csv, torch, npy)

Pose.txt values are in mm

Keypoint detection and matching is facilitated by the from image_matching_models.matching import get_matcher