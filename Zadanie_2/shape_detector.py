import cv2 as cv
import numpy as np
import os
import glob

from ximea import xiapi

config = {
    "use_hsv": True,
    "swap_hsv_colors": True,
    "old_color": [60, 200, 200],
    "new_color": [120, 255, 255]
}

def capture_webcam(cap=None, save_dir=None, prefix='capture', preview=False):
    own_cap = False
    if cap is None:
        cap = cv.VideoCapture(0)
        own_cap = True

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    frames = []

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(save_dir, exist_ok=True)

    existing = glob.glob(os.path.join(save_dir, f"{prefix}_*.jpg"))
    start_idx = 0
    for e in existing:
        name = os.path.splitext(os.path.basename(e))[0]
        try:
            idx = int(name.rsplit('_', 1)[1])
            if idx > start_idx:
                start_idx = idx
        except Exception:
            pass
    save_idx = start_idx + 1

    if preview:
        cv.namedWindow('Preview', cv.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv.imshow('Preview', frame)
            k = cv.waitKey(30) & 0xFF
            if k == 32:  # SPACE
                fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
                cv.imwrite(fname, frame)
                frames.append(frame)
                save_idx += 1
                print(f"Saved {len(frames)} images to {save_dir}")
            elif k == ord('q'):
                break
        cv.destroyWindow('Preview')
    else:
        ret, frame = cap.read()
        if ret:
            fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
            cv.imwrite(fname, frame)
            frames.append(frame)

    if own_cap:
        cap.release()

    return frames

def capture_ximea(cam=None, save_dir=None, prefix='capture', preview=False):
    own_cam = False

    if cam is None:
        cam = xiapi.Camera()
        cam.open_device()
        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)
        cam.start_acquisition()
        own_cam = True

    img = xiapi.Image()
    frames = []

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(save_dir, exist_ok=True)

    existing = glob.glob(os.path.join(save_dir, f"{prefix}_*.jpg"))
    start_idx = 0
    for e in existing:
        name = os.path.splitext(os.path.basename(e))[0]
        try:
            idx = int(name.rsplit('_', 1)[1])
            if idx > start_idx:
                start_idx = idx
        except Exception:
            pass
    save_idx = start_idx + 1

    if preview:
        cv.namedWindow('Preview', cv.WINDOW_NORMAL)
        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            if frame is None:
                print("Failed to grab frame")
                break
            
            frame = cv.resize(frame, (1280, 720))

            cv.imshow('Preview', frame)
            k = cv.waitKey(30) & 0xFF

            if k == 32:  # SPACE
                fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
                cv.imwrite(fname, frame)
                frames.append(frame)
                save_idx += 1
                print(f"Saved {len(frames)} images to {save_dir}")

            elif k == ord('q'):
                break

        cv.destroyWindow('Preview')

    else:
        cam.get_image(img)
        frame = img.get_image_data_numpy()

        if frame is not None:
            frame = cv.resize(frame, (1280, 720))
            fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
            cv.imwrite(fname, frame)
            frames.append(frame)

    if own_cam:
        cam.stop_acquisition()
        cam.close_device()

    return frames

def detect_shapes_in_frame(frame, config=None):
    if config is None:
        config = {}

    image = frame.copy()

    use_hsv = config.get("use_hsv", False)
    swap_hsv_colors = config.get("swap_hsv_colors", False)
    old_color = config.get("old_color")
    new_color = config.get("new_color")

    if use_hsv:
        hsv = frame_to_hsv(image)

        if swap_hsv_colors:
            if old_color is None or new_color is None:
                raise ValueError("old_color and new_color must be provided when swap_hsv_colors=True")
            
            #print(f"Swapping color {old_color} to {new_color} in HSV space")
            hsv = swap_color(hsv, old_color, new_color)

        image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    output = image.copy()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)

    thresh_image = cv.Canny(blurred, 20, 60)

    contours, _ = cv.findContours(
        thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    circels = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=120,
        param1=150,
        param2=50,
        minRadius=20,
        maxRadius=80)

    if circels is not None:
        
        circels = np.round(circels[0, :]).astype("int")

        for (x, y, r) in circels:
            cv.circle(output, (x, y), r, (0, 255, 255), 3)
            cv.putText(
                output,
                "Circle",
                (x - r, y - r - 10),
                cv.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    for contour in contours:

        area = cv.contourArea(contour)
        if area < 1500:
            continue

        peri = cv.arcLength(contour, True)
        if peri == 0:
            continue

        epsilon = 0.04 * peri
        approx = cv.approxPolyDP(contour, epsilon, True)
        if not cv.isContourConvex(approx):
            continue

        x, y, w, h = cv.boundingRect(approx)
        if h == 0:
            continue
        ratio = w / float(h)

        vertices = len(approx)
        shape = "Unknown"

        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:

            if abs(1.0 - ratio) <= 0.2:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif vertices == 5:
            shape = "Pentagon"
            
        # else:
        #     circularity = 4 * np.pi * area / (peri * peri)
        #     if circularity > 0.8:
        #         shape = "Circle"

        if shape != "Unknown":

            cv.drawContours(output, [approx], -1, (0, 255, 0), 3)

            M = cv.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv.circle(output, (cx, cy), 5, (255, 0, 0), -1)

                cv.putText(
                    output,
                    shape,
                    (x, y - 10),
                    cv.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

    return output, thresh_image

def detect_shapes_webcam(save_dir=None, prefix='capture', config=None):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        frames = capture_webcam(cap=cap, save_dir=save_dir, prefix=prefix, preview=False)

        if not frames:
            print("No frame captured")
            break

        frame = frames[0]
        detected, thresh = detect_shapes_in_frame(frame, config=config)

        cv.imshow("Threshold", thresh)
        cv.imshow("Shape detection", detected)

        k = cv.waitKey(30) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def detect_shapes_ximea(save_dir=None, prefix='capture', config=None):
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_exposure(50000)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)

    cam.start_acquisition()
    img = xiapi.Image()

    while True:

        cam.get_image(img)
        frame = img.get_image_data_numpy()

        if frame is None:
            print("No frame captured")
            break

        frame = cv.resize(frame, (640, 480))

        detected, thresh = detect_shapes_in_frame(frame, config=config)

        cv.imshow("Threshold", thresh)
        cv.imshow("Shape detection", detected)

        k = cv.waitKey(30) & 0xFF
        if k == ord('q'):
            break

    cam.stop_acquisition()
    cam.close_device()
    cv.destroyAllWindows()

def frame_to_hsv(frame):
    if frame is None:
        raise ValueError("Input frame is None")

    if len(frame.shape) != 3:
        raise ValueError("Input frame must be a color image")

    channels = frame.shape[2]

    if channels == 3:
        bgr = frame
    elif channels == 4:
        bgr = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    return hsv


def swap_color(hsv, old_color, new_color, tol_h=10, tol_s=120, tol_v=120):
    if hsv is None:
        raise ValueError("Input HSV image is None")

    if len(hsv.shape) != 3 or hsv.shape[2] != 3:
        raise ValueError("Input HSV image must have 3 channels")

    result = hsv.copy()

    old_color = np.array(old_color, dtype=np.int16)
    new_color = np.array(new_color, dtype=np.uint8)

    lower = np.array([
        max(0, old_color[0] - tol_h),
        max(0, old_color[1] - tol_s),
        max(0, old_color[2] - tol_v)
    ], dtype=np.uint8)

    upper = np.array([
        min(179, old_color[0] + tol_h),
        min(255, old_color[1] + tol_s),
        min(255, old_color[2] + tol_v)
    ], dtype=np.uint8)

    mask = cv.inRange(result, lower, upper)
    result[mask > 0] = new_color

    return result


    


if __name__ == "__main__":
    #detect_shapes_webcam()
    config = {
    "use_hsv": True,
    "swap_hsv_colors": True,
    "old_color": [255, 255, 0],
    "new_color": [0, 255, 255]
}

detect_shapes_ximea(config=config)