"""
Docstring for Zadanie_1.main

Aby spustit webku: py .\Zadanie_1\main.py --camera webcam
Aby spustit ximea: py .\Zadanie_1\main.py --camera ximea
"""



from ximea import xiapi
import cv2
import numpy as np
import argparse
import sys

NUM_IMAGES = 4
RESIZE_LEN = 256
RESIZE = (RESIZE_LEN, RESIZE_LEN)

def capture_ximea(resize = RESIZE):

    cam = xiapi.Camera()
    cam.open_device()
    cam.set_exposure(50000) 
    cam.set_param("imgdataformat","XI_RGB32")
    cam.set_param("auto_wb",1)
    img = xiapi.Image()
    cam.start_acquisition()

    frames = []

    for i in range(NUM_IMAGES):
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image, resize)
        frames.append(image)

    cam.stop_acquisition()
    cam.close_device()

    return frames

def capture_webcam(resize = RESIZE, device=0):

    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Could not open video device")

    frames = []

    for i in range(NUM_IMAGES):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {i} from webcam")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv BGR -> ximea RGB
        frame = cv2.resize(frame, resize)
        frames.append(frame)
        print(f"Captured frame {i} from webcam")

    cap.release()

    return frames

def make_mozaic_and_process(frames):

    assert len(frames) == NUM_IMAGES, f"Expected {NUM_IMAGES} frames, got {len(frames)}"

    top_row = cv2.hconcat(frames[0:2])
    bottom_row = cv2.hconcat(frames[2:4])
    mozaic = cv2.vconcat([top_row, bottom_row])

    mozaic = mozaic[:,:, 0:3] # remove alpha channel if present

    #====== 2 - Sobel filter =======

    mozaic[0:RESIZE_LEN, RESIZE_LEN:2*RESIZE_LEN] = cv2.cvtColor(
                            cv2.cvtColor(mozaic[0:RESIZE_LEN, RESIZE_LEN:2*RESIZE_LEN],
                            cv2.COLOR_RGB2GRAY), 
                            cv2.COLOR_GRAY2BGR)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    # apply sobel image filter on x and y direction
    gx = cv2.filter2D(mozaic[0:RESIZE_LEN, RESIZE_LEN:2*RESIZE_LEN], cv2.CV_64F, sobel_x)
    gy = cv2.filter2D(mozaic[0:RESIZE_LEN, RESIZE_LEN:2*RESIZE_LEN], cv2.CV_64F, sobel_y)
    
    # calculate magnitude of gradient
    magnitude = np.sqrt(gx**2 + gy**2)

    # normalize magnitude to range [0, 255] and convert to uint8
    magnitude_grayscale = np.mean(magnitude, axis=2)
    magnitude_normalized = cv2.normalize(magnitude_grayscale, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # convert back to RGB and place in mozaic
    mozaic[0:RESIZE_LEN, RESIZE_LEN:2*RESIZE_LEN] = cv2.cvtColor(magnitude_normalized, cv2.COLOR_GRAY2RGB)


    #====== 3 - 90 degree rotation =======

    rotated_image = mozaic[RESIZE_LEN:2*RESIZE_LEN, 0:RESIZE_LEN].copy()
    rotated_image = np.zeros(rotated_image.shape, dtype=rotated_image.dtype)    
    for i in range(rotated_image.shape[0]):
        for j in range(rotated_image.shape[1]):
            rotated_image[j, rotated_image.shape[0]-1-i] = mozaic[RESIZE_LEN:2*RESIZE_LEN, 0:RESIZE_LEN][i, j]

    mozaic[RESIZE_LEN:2*RESIZE_LEN, 0:RESIZE_LEN] = rotated_image

    #====== 4 - Red channel only =======

    mozaic[RESIZE_LEN:2*RESIZE_LEN, RESIZE_LEN:2*RESIZE_LEN, 0:2] = 0

    return mozaic

def main():
    parser = argparse.ArgumentParser(description="Choose camera source")
    parser.add_argument('-c', '--camera', choices=['webcam', 'ximea'], default='webcam',
                        help='Camera to use (webcam or ximea)')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Webcam device index (used when --camera=webcam)')
    args = parser.parse_args()

    if args.camera == 'webcam':
        cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Cannot open webcam {args.index}", file=sys.stderr)
            sys.exit(1)

        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        print("Preview running. Press SPACE to snap images, 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam", file=sys.stderr)
                break
            cv2.imshow("Preview", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:  # space
                frames = []
                for _ in range(NUM_IMAGES):
                    ret, f = cap.read()
                    if not ret:
                        break
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    f = cv2.resize(f, RESIZE)
                    frames.append(f)
                if len(frames) == NUM_IMAGES:
                    mozaic = make_mozaic_and_process(frames)
                    cv2.imshow("Mozaic", mozaic)
                    print("Mozaic shown. Press SPACE to capture again or 'q' to quit.")
                else:
                    print("Could not capture required frames", file=sys.stderr)
            elif k == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:  # ximea
        cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
        print("Ximea mode. Press SPACE in the window to snap images, 'q' to quit.")
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == 32:  # space
                try:
                    frames = capture_ximea()
                    if len(frames) == NUM_IMAGES:
                        mozaic = make_mozaic_and_process(frames)
                        cv2.imshow("Mozaic", mozaic)
                        print("Mozaic shown. Press SPACE to capture again or 'q' to quit.")
                    else:
                        print("Ximea returned insufficient frames", file=sys.stderr)
                except Exception as e:
                    print(f"Ximea error: {e}", file=sys.stderr)
            elif k == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()