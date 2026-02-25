import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./Zadanie_2/images/*.jpg')

print(len(images))

for fname in images:
    img = cv.imread(fname)

    if img is None:
        print(f"Failed to read image {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(f"Processing {fname} - image shape: {img.shape}, gray shape: {gray.shape}")


    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)

    print(f'ret: {ret}, fname: {fname}')

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        #cv.imshow('img', img)
        cv.imwrite('./Zadanie_2/Cada_corners_found.jpg', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.set_printoptions(precision=4, suppress=True)

print(f"Re-projection RMS error: {ret:.6f}\n")

print("Camera matrix (intrinsics):")
print(mtx, "\n")

print("Distortion coefficients (flattened):")
print(dist.ravel(), "\n")

print("Extrinsic parameters (per image):")
for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs), start=1):
    r = np.squeeze(rvec)
    t = np.squeeze(tvec)
    print(f"Image {i:02d}:")
    print(f"  rvec: {r}")
    print(f"  tvec: {t}\n")

