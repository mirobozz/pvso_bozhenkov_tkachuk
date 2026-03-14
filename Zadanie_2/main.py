import numpy as np
import cv2 as cv
import glob
import argparse
import sys
import os

# remove top-level capture_ximea() call and add main with preview/snapping
def main():
    # print("Ximea preview. Press SPACE to snap images, 'q' to quit.")
    # capture_ximea(preview=True)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob('*.jpg')
    #images = glob.glob('Zadanie_2/images/capture_016.jpg')


    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,5), corners2, ret)
            cv.imshow('img', img)

            cv.imwrite('corners_found.jpg', img)

            cv.waitKey(500)

    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()