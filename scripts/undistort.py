# import os
# import cv2
# import numpy as np
# import argparse
# from glob import glob


# def undistort_and_save(video_name, input_dir='data', output_dir='data'):
#     video_path = os.path.join(input_dir, video_name)
#     intrinsics_path = os.path.join(video_path, 'intrinsics.npy')
#     dist_coeffs_path = os.path.join(video_path, 'dist_coeffs.npy')

#     if not os.path.exists(intrinsics_path) or not os.path.exists(dist_coeffs_path):
#         raise FileNotFoundError("Missing intrinsics or distortion coefficients")

#     K = np.load(intrinsics_path)
#     D = np.load(dist_coeffs_path)

#     frame_paths = sorted(glob(os.path.join(video_path, 'frames', '*.png')))
#     depth_paths = sorted(glob(os.path.join(video_path, 'depths', '*.png')))

#     out_img_dir = os.path.join(output_dir, video_name, 'undistorted_frames')
#     out_depth_dir = os.path.join(output_dir, video_name, 'undistorted_depths')
#     os.makedirs(out_img_dir, exist_ok=True)
#     os.makedirs(out_depth_dir, exist_ok=True)

#     for i, (fpath, dpath) in enumerate(zip(frame_paths, depth_paths)):
#         img = cv2.imread(fpath)
#         depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)

#         h, w = img.shape[:2]
#         new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
#         map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)

#         img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
#         depth_undistorted = cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST)

#         img_name = os.path.basename(fpath)
#         depth_name = os.path.basename(dpath)

#         cv2.imwrite(os.path.join(out_img_dir, img_name), img_undistorted)
#         cv2.imwrite(os.path.join(out_depth_dir, depth_name), depth_undistorted)

#         if i % 20 == 0:
#             print(f"Processed {i}/{len(frame_paths)} frames")

#     print("Undistortion complete.")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video_name', required=True)
#     args = parser.parse_args()

#     undistort_and_save(args.video_name)
