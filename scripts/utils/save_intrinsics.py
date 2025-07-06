import numpy as np
import os
import argparse


def save_intrinsics_and_distortion(video_name, out_dir=None):
    if out_dir is None:
        out_dir = os.path.abspath("/workspace/honor/my_project/data/C3VD")


    fx = 767.3861511125845
    fy = 767.5058656118406
    cx = 679.054265997005
    cy = 543.646891684636
    k1 = -0.18867185058223412
    k2 = -0.003927337093919806
    k3 = 0.030524814153620117
    k4 = -0.012756926010904904

    # Set camera intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # # Set distortion coefficients - for fisheye camera model
    D = np.array([k1, k2, k3, k4])

    video_path = os.path.join(out_dir, video_name)
    os.makedirs(video_path, exist_ok=True)

    np.save(os.path.join(video_path, "intrinsics.npy"), K)
    np.save(os.path.join(video_path, "dist_coeffs.npy"), D)

    print(f"Saved intrinsics and distortion to: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", required=True, help="Video name to save under data/<video_name>/")
    args = parser.parse_args()

    save_intrinsics_and_distortion(args.video_name)


#python save_intrinsics.py --video_name my_video_name
