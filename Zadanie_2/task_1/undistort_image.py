import argparse
import json

import cv2 as cv
import numpy as np


DEFAULT_PARAMS_JSON = "./camera_params.json"


def main():
    parser = argparse.ArgumentParser(description="Undistort an image using saved camera parameters.")
    parser.add_argument("input_image", help="Path to input image.")
    parser.add_argument(
        "--params",
        default=DEFAULT_PARAMS_JSON,
        help=f"Path to JSON file with camera parameters (default: {DEFAULT_PARAMS_JSON}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save undistorted image (default: undistorted_<input_name>).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop the undistorted image to ROI.",
    )
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = json.load(f)

    camera_matrix = np.array(params["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(params["dist_coeffs"], dtype=np.float64)

    img = cv.imread(args.input_image)
    if img is None:
        raise FileNotFoundError(f"Failed to read input image: {args.input_image}")

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    if not args.no_crop:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y : y + rh, x : x + rw]

    output_path = args.output
    if output_path is None:
        output_path = args.input_image.replace("\\", "/").split("/")[-1]
        output_path = f"undistorted_images/undistorted_{output_path}"

    cv.imwrite(output_path, undistorted)
    print(f"Undistorted image saved to {output_path}")


if __name__ == "__main__":
    main()
