import glob
import json
import cv2 as cv
import numpy as np
import os


CHESSBOARD_SIZE = (7, 5)
IMAGES_GLOB = "./images/*.jpg"
OUTPUT_JSON = "./camera_params.json" 


def main():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(IMAGES_GLOB)
    print(f"Found {len(images)} images")

    gray_shape = None
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Failed to read image {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        print(f"Processing {fname}: corners found = {ret}")

        if not ret:
            continue

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

    if not objpoints or gray_shape is None:
        raise RuntimeError("No valid chessboard detections found. Calibration cannot be computed.")

    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    mean_error = 0.0
    for i in range(len(objpoints)):
        projected, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], projected, cv.NORM_L2) / len(projected)
        mean_error += error
    mean_error /= len(objpoints)

    data = {
        "chessboard_size": list(CHESSBOARD_SIZE),
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error": float(mean_error),
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    np.set_printoptions(precision=4, suppress=True)
    print(f"Saved camera parameters to {OUTPUT_JSON}")
    print(f"RMS re-projection error: {rms:.6f}")
    print(f"Mean re-projection error: {mean_error:.6f}")
    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist.ravel())


if __name__ == "__main__":
    main()

    
