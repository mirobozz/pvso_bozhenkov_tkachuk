import cv2 as cv
import numpy as np
import os
import glob

from ximea import xiapi

def _resize_keep_aspect(frame, scale=0.5):
    if frame is None:
        return frame
    if scale <= 0:
        raise ValueError("scale must be > 0")
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

def nothing(_):
    pass

def _rgb_to_hsv_pixel(rgb):
    arr = np.uint8([[rgb]])
    return cv.cvtColor(arr, cv.COLOR_RGB2HSV)[0, 0]

def _hsv_to_rgb_pixel(hsv):
    arr = np.uint8([[hsv]])
    return cv.cvtColor(arr, cv.COLOR_HSV2RGB)[0, 0]

def init_hsv_trackbars(config, window_name="HSV Controls"):
    old_rgb = config.get("old_color", [225, 180, 150])
    new_rgb = config.get("new_color", [210, 35, 35])

    old_hsv = _rgb_to_hsv_pixel(old_rgb)
    new_hsv = _rgb_to_hsv_pixel(new_rgb)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    # old color
    cv.createTrackbar("From H", window_name, int(old_hsv[0]), 179, nothing)
    cv.createTrackbar("From S", window_name, int(old_hsv[1]), 255, nothing)
    cv.createTrackbar("From V", window_name, int(old_hsv[2]), 255, nothing)

    # target color 
    cv.createTrackbar("To H", window_name, int(new_hsv[0]), 179, nothing)
    cv.createTrackbar("To S", window_name, int(new_hsv[1]), 255, nothing)
    cv.createTrackbar("To V", window_name, int(new_hsv[2]), 255, nothing)


def update_colors_from_hsv_trackbars(config, window_name="HSV Controls"):
    from_h = cv.getTrackbarPos("From H", window_name)
    from_s = cv.getTrackbarPos("From S", window_name)
    from_v = cv.getTrackbarPos("From V", window_name)

    to_h = cv.getTrackbarPos("To H", window_name)
    to_s = cv.getTrackbarPos("To S", window_name)
    to_v = cv.getTrackbarPos("To V", window_name)

    old_hsv = np.array([from_h, from_s, from_v], dtype=np.uint8)
    new_hsv = np.array([to_h, to_s, to_v], dtype=np.uint8)

    config["old_color"] = _hsv_to_rgb_pixel(old_hsv).tolist()
    config["new_color"] = _hsv_to_rgb_pixel(new_hsv).tolist()


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

def init_canny_trackbars(config, window_name="Canny Controls"):
    t1 = int(config.get("canny_threshold1", 10))
    t2 = int(config.get("canny_threshold2", 40))

    t1 = max(0, min(255, t1))
    t2 = max(0, min(255, t2))

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.createTrackbar("Canny T1", window_name, t1, 255, nothing)
    cv.createTrackbar("Canny T2", window_name, t2, 255, nothing)


def update_canny_from_trackbars(config, window_name="Canny Controls"):
    t1 = cv.getTrackbarPos("Canny T1", window_name)
    t2 = cv.getTrackbarPos("Canny T2", window_name)

    # keep thresholds ordered
    if t1 > t2:
        t1, t2 = t2, t1
        cv.setTrackbarPos("Canny T1", window_name, t1)
        cv.setTrackbarPos("Canny T2", window_name, t2)

    config["canny_threshold1"] = int(t1)
    config["canny_threshold2"] = int(t2)

def detect_shapes_in_frame(frame, config=None, allow_size_measurement=False):
    if config is None:
        config = {}

    image = frame.copy()

    # ximeaa 4-channel images -> 3-channel
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    swap_rgb_colors = config.get("swap_rgb_colors", False)
    old_color = config.get("old_color")
    new_color = config.get("new_color")

    requested_measure_size = config.get("measure_size", False)
    measure_size = bool(allow_size_measurement and requested_measure_size)

    distance_cm = config.get("distance_cm")
    camera_matrix = config.get("camera_matrix")
    dist_coeffs = config.get("dist_coeffs")

    if measure_size:
        if camera_matrix is None or dist_coeffs is None:
            raise ValueError("camera_matrix and dist_coeffs must be provided when measure_size=True")
        if distance_cm is None:
            raise ValueError("distance_cm must be provided when measure_size=True")
        image = undistort_frame(image, camera_matrix, dist_coeffs)

    # detection image 
    analysis_image = image

    # display image 
    display_image = analysis_image.copy()
    if swap_rgb_colors:
        if old_color is None or new_color is None:
            raise ValueError("old_color and new_color must be provided when swap_rgb_colors=True")

        rgb = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
        rgb = swap_color(rgb, old_color, new_color)
        display_image = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    output = display_image.copy()

    # detect from original analysis image only
    gray = cv.cvtColor(analysis_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)

    canny_t1 = int(config.get("canny_threshold1", 10))
    canny_t2 = int(config.get("canny_threshold2", 40))
    canny_t1 = max(0, min(255, canny_t1))
    canny_t2 = max(0, min(255, canny_t2))
    if canny_t1 > canny_t2:
        canny_t1, canny_t2 = canny_t2, canny_t1

    thresh_image = cv.Canny(blurred, canny_t1, canny_t2)

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
            cv.putText(output, "Circle", (x - r, y - r - 10),
                       cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

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
            shape = "Square" if abs(1.0 - ratio) <= 0.2 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"

        if shape != "Unknown":
            cv.drawContours(output, [approx], -1, (0, 255, 0), 3)
            M = cv.moments(contour)
            label = shape

            if measure_size:
                size_info = measure_contour_size_cm(contour, camera_matrix, distance_cm)
                label = f"{shape} {size_info['width_cm']:.1f}x{size_info['height_cm']:.1f} cm"

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(output, (cx, cy), 5, (255, 0, 0), -1)
                cv.putText(output, label, (x, y - 10),
                           cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

    return output, thresh_image

def detect_shapes_webcam(save_dir=None, prefix='capture', config=None):
    if config is None:
        config = {}

    init_hsv_trackbars(config, "HSV Controls")
    init_canny_trackbars(config, "Canny Controls")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        update_colors_from_hsv_trackbars(config, "HSV Controls")
        update_canny_from_trackbars(config, "Canny Controls")
        frames = capture_webcam(cap=cap, save_dir=save_dir, prefix=prefix, preview=False)

        if not frames:
            print("No frame captured")
            break

        frame = frames[0]
        detected, thresh = detect_shapes_in_frame(
            frame,
            config=config,
            allow_size_measurement=False,  
        )

        cv.imshow("Threshold", thresh)
        cv.imshow("Shape detection", detected)

        k = cv.waitKey(30) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def detect_shapes_ximea(save_dir=None, prefix='capture', config=None):
    if config is None:
        config = {}

    init_hsv_trackbars(config, "HSV Controls")
    init_canny_trackbars(config, "Canny Controls")

    cam = xiapi.Camera()
    cam.open_device()
    cam.set_exposure(50000)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)

    cam.start_acquisition()
    img = xiapi.Image()

    display_scale = config.get("ximea_display_scale", 0.5)  # smaller window, same proportions

    while True:
        update_colors_from_hsv_trackbars(config, "HSV Controls")
        update_canny_from_trackbars(config, "Canny Controls")
        cam.get_image(img)
        frame = img.get_image_data_numpy()

        if frame is None:
            print("No frame captured")
            break

        # keep original frame for detection/measurement (no distortion)
        detected, thresh = detect_shapes_in_frame(
            frame,
            config=config,
            allow_size_measurement=True,
        )

        # resize only for visualization
        detected_view = _resize_keep_aspect(detected, display_scale)
        thresh_view = _resize_keep_aspect(thresh, display_scale)

        cv.imshow("Threshold", thresh_view)
        cv.imshow("Shape detection", detected_view)

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

def swap_color(
    rgb,
    old_color,
    new_color,
    tol_h=10,
    tol_s=40,
    tol_v=40,
    preserve_sv=False,
):

    if rgb is None:
        raise ValueError("Input RGB image is None")

    if not isinstance(rgb, np.ndarray):
        raise TypeError("Input RGB image must be a numpy array")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Input RGB image must have shape (H, W, 3)")

    old_color = np.asarray(old_color, dtype=np.uint8)
    new_color = np.asarray(new_color, dtype=np.uint8)

    if old_color.shape != (3,):
        raise ValueError("old_color must contain exactly 3 values: [R, G, B]")
    if new_color.shape != (3,):
        raise ValueError("new_color must contain exactly 3 values: [R, G, B]")

    for name, color in (("old_color", old_color), ("new_color", new_color)):
        if np.any(color < 0) or np.any(color > 255):
            raise ValueError(f"{name} values must be in range 0..255")

    for name, tol, max_value in (("tol_h", tol_h, 179), ("tol_s", tol_s, 255), ("tol_v", tol_v, 255)):
        if not isinstance(tol, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if tol < 0 or tol > max_value:
            raise ValueError(f"{name} must be in range 0..{max_value}")

    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

    old_hsv = cv.cvtColor(old_color.reshape(1, 1, 3), cv.COLOR_RGB2HSV)[0, 0].astype(np.int16)
    new_hsv = cv.cvtColor(new_color.reshape(1, 1, 3), cv.COLOR_RGB2HSV)[0, 0].astype(np.uint8)

    h, s, v = map(int, old_hsv)

    s_low = max(0, s - tol_s)
    s_high = min(255, s + tol_s)
    v_low = max(0, v - tol_v)
    v_high = min(255, v + tol_v)

    h_low = h - tol_h
    h_high = h + tol_h

    if h_low < 0:
        lower1 = np.array([0, s_low, v_low], dtype=np.uint8)
        upper1 = np.array([h_high, s_high, v_high], dtype=np.uint8)
        lower2 = np.array([180 + h_low, s_low, v_low], dtype=np.uint8)
        upper2 = np.array([179, s_high, v_high], dtype=np.uint8)
        mask = cv.inRange(hsv, lower1, upper1) | cv.inRange(hsv, lower2, upper2)
    elif h_high > 179:
        lower1 = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper1 = np.array([179, s_high, v_high], dtype=np.uint8)
        lower2 = np.array([0, s_low, v_low], dtype=np.uint8)
        upper2 = np.array([h_high - 180, s_high, v_high], dtype=np.uint8)
        mask = cv.inRange(hsv, lower1, upper1) | cv.inRange(hsv, lower2, upper2)
    else:
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv.inRange(hsv, lower, upper)

    result_hsv = hsv.copy()
    if preserve_sv:
        result_hsv[..., 0][mask > 0] = new_hsv[0]
    else:
        result_hsv[mask > 0] = new_hsv

    result_rgb = cv.cvtColor(result_hsv, cv.COLOR_HSV2RGB)
    return result_rgb

def undistort_frame(frame, camera_matrix, dist_coeffs):
    if frame is None:
        raise ValueError("frame is None")

    if camera_matrix is None or dist_coeffs is None:
        raise ValueError("camera_matrix and dist_coeffs are required")

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    return cv.undistort(frame, camera_matrix, dist_coeffs)

def measure_contour_size_cm(contour, camera_matrix, distance_cm):

    if contour is None:
        raise ValueError("contour is None")

    if camera_matrix is None:
        raise ValueError("camera_matrix is required")

    if distance_cm <= 0:
        raise ValueError("distance_cm must be > 0")

    fx = float(camera_matrix[0][0])
    fy = float(camera_matrix[1][1])

    rect = cv.minAreaRect(contour)
    width_px, height_px = rect[1]

    width_cm = (width_px * distance_cm) / fx
    height_cm = (height_px * distance_cm) / fy

    return {
        "width_px": width_px,
        "height_px": height_px,
        "width_cm": width_cm,
        "height_cm": height_cm,
        "rect": rect,
    }



if __name__ == "__main__":

    config = {
    "use_hsv": True,
    "swap_rgb_colors": True,
    "old_color": [35, 35, 210],
    "new_color": [210, 35, 35],

    "canny_threshold1": 10,
    "canny_threshold2": 40,

    "measure_size": True,
    "distance_cm": 40.0,
    "camera_matrix": [
        [3728.7181333912044, 0.0, 1270.7614108415155],
        [0.0, 3724.522617566526, 994.1047999698993],
        [0.0, 0.0, 1.0]
    ],
    "dist_coeffs": [[
        -0.35471040526886033,
        -0.6906196550551162,
        0.000179167973734459,
        -0.00220198755170272,
        5.548085386889819
    ]],

    "ximea_display_scale": 0.25,
}

    #detect_shapes_ximea(config=config)
    detect_shapes_webcam(config=config)
