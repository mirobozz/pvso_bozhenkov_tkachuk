import argparse
import sys

import cv2 as cv
import numpy as np

try:
    from ximea import xiapi
except ImportError:
    xiapi = None


WINDOW_NAME = "VideoFlow"
DEFAULT_INPUT_PATH = "glupik.jpg"
DISPLAY_TILE_SIZE = (480, 360)
GLOBAL_THRESHOLD = 128
ADAPTIVE_WINDOW_SIZE = 31
ADAPTIVE_T_PERCENT = 7.5


def load_image(path):
    try:
        return cv.imread(path, cv.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None


def save_image(path, image):
    try:
        cv.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")


def rbga_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image [:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (BGR)")

    b = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    r = image[:, :, 2].astype(np.float64)

    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def global_thresholding_gray(grayscale: np.ndarray, threshold: int) -> np.ndarray:
    return np.where(grayscale >= threshold, 255, 0).astype(np.uint8)


def global_thresholding(image: np.ndarray, threshold: int) -> np.ndarray:
    grayscale = rgb_to_grayscale(image)
    return global_thresholding_gray(grayscale, threshold)


def otsu_thresholding_gray(grayscale: np.ndarray) -> np.ndarray:
    grayscale = grayscale.astype(np.uint8)

    hist = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()

    bins = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bins)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b2 = (mu_total * omega - mu) ** 2 / (denom + 1e-12)
    sigma_b2[(omega <= 0) | (omega >= 1)] = -1

    threshold_signed = int(np.argmax(sigma_b2))
    return global_thresholding_gray(grayscale, threshold_signed)


def otsu_thresholding(image: np.ndarray, threshold: int = 0) -> np.ndarray:
    grayscale = rgb_to_grayscale(image).astype(np.uint8)
    return otsu_thresholding_gray(grayscale)


def adaptive_mean_thresholding_percent(image: np.ndarray, window_size: int, t_percent: float) -> np.ndarray:
    grayscale = rgb_to_grayscale(image)
    return adaptive_mean_thresholding_percent_gray(grayscale, window_size, t_percent)


def adaptive_mean_thresholding_percent_gray(
    grayscale: np.ndarray,
    window_size: int,
    t_percent: float,
) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")
    if not (0 <= t_percent <= 100):
        raise ValueError("T percent must be in the range [0, 100].")

    height, width = grayscale.shape
    half = window_size // 2

    integral = np.pad(grayscale.astype(np.float64), ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    y = np.arange(height)
    x = np.arange(width)

    y1 = np.clip(y - half, 0, height - 1)
    y2 = np.clip(y + half, 0, height - 1)
    x1 = np.clip(x - half, 0, width - 1)
    x2 = np.clip(x + half, 0, width - 1)

    top = y1[:, None]
    bottom = (y2 + 1)[:, None]
    left = x1[None, :]
    right = (x2 + 1)[None, :]

    window_sum = (
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )

    counts = (y2 - y1 + 1)[:, None] * (x2 - x1 + 1)[None, :]
    local_mean = window_sum / counts
    threshold = local_mean * (100.0 - t_percent) / 100.0

    return np.where(grayscale <= threshold, 0, 255).astype(np.uint8)


def resize_for_tile(image: np.ndarray) -> np.ndarray:
    return cv.resize(image, DISPLAY_TILE_SIZE, interpolation=cv.INTER_AREA)


def threshold_to_bgr(threshold_image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(threshold_image, cv.COLOR_GRAY2BGR)


def annotate_tile(image: np.ndarray, label: str) -> np.ndarray:
    annotated = image.copy()
    cv.putText(
        annotated,
        label,
        (12, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )
    return annotated


def build_video_panel(frame: np.ndarray) -> np.ndarray:
    frame = bgr_to_rgb(frame)
    frame = resize_for_tile(frame)
    
    grayscale = rgb_to_grayscale(frame)

    global_binary = global_thresholding_gray(grayscale, GLOBAL_THRESHOLD)
    otsu_binary = otsu_thresholding_gray(grayscale)
    adaptive_binary = adaptive_mean_thresholding_percent_gray(
        grayscale,
        window_size=ADAPTIVE_WINDOW_SIZE,
        t_percent=ADAPTIVE_T_PERCENT,
    )

    original_tile = annotate_tile(frame, "Original")
    global_tile = annotate_tile(threshold_to_bgr(global_binary), "Global")
    otsu_tile = annotate_tile(threshold_to_bgr(otsu_binary), "Otsu")
    adaptive_tile = annotate_tile(threshold_to_bgr(adaptive_binary), "Adaptive")

    top_row = cv.hconcat([original_tile, global_tile])
    bottom_row = cv.hconcat([otsu_tile, adaptive_tile])
    return cv.vconcat([top_row, bottom_row])


def setup_ximea_camera():
    if xiapi is None:
        raise ImportError("ximea package is not installed, so Ximea mode is unavailable.")

    try:
        print("Initializing Ximea camera...")
        cam = xiapi.Camera()
        cam.open_device()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Ximea camera: {e}")
    try:

        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)
    except Exception as e:
        raise RuntimeError(f"Failed to configure Ximea camera: {e}")
    

    img = xiapi.Image()

    try:
        cam.start_acquisition()
    except Exception as e:
        raise RuntimeError(f"Failed to start Ximea camera acquisition: {e}")
    return cam, img


def read_ximea_frame(cam, img):
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = rbga_to_rgb(frame)
    return cv.cvtColor(frame, cv.COLOR_RGB2BGR)


def run_webcam_flow(device: int):
    cap = cv.VideoCapture(device, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {device}")

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, DISPLAY_TILE_SIZE[0] * 2, DISPLAY_TILE_SIZE[1] * 2)
    print("Webcam VideoFlow running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam.")

            frame = cv.flip(frame, 1)
            panel = build_video_panel(frame)
            cv.imshow(WINDOW_NAME, panel)

            if (cv.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def run_ximea_flow():
    cam, img = setup_ximea_camera()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, DISPLAY_TILE_SIZE[0] * 2, DISPLAY_TILE_SIZE[1] * 2)
    print("Ximea VideoFlow running. Press 'q' to quit.")

    try:
        while True:
            frame = read_ximea_frame(cam, img)
            panel = build_video_panel(frame)
            cv.imshow(WINDOW_NAME, panel)

            if (cv.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cam.stop_acquisition()
        cam.close_device()
        cv.destroyAllWindows()


def process_image_file(input_path: str, output_prefix: str):
    image = load_image(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    image = rbga_to_rgb(image)
    global_image = global_thresholding(image, GLOBAL_THRESHOLD)
    otsu_image = otsu_thresholding(image)
    adaptive_image = adaptive_mean_thresholding_percent(
        image,
        window_size=ADAPTIVE_WINDOW_SIZE,
        t_percent=ADAPTIVE_T_PERCENT,
    )
    panel = build_video_panel(image)

    save_image(f"{output_prefix}_global.png", global_image)
    save_image(f"{output_prefix}_otsu.png", otsu_image)
    save_image(f"{output_prefix}_adaptive.png", adaptive_image)
    save_image(f"{output_prefix}_panel.png", panel)
    print(f"Saved outputs with prefix '{output_prefix}'.")


def main():
    parser = argparse.ArgumentParser(description="Thresholding for images, webcam, or Ximea camera.")
    parser.add_argument(
        "-c",
        "--camera",
        choices=["webcam", "ximea"],
        help="Run the live VideoFlow window using the selected camera source.",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Webcam device index (used when --camera=webcam).",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Input image path used when no live camera is selected.",
    )
    parser.add_argument(
        "--output-prefix",
        default="glupik",
        help="Output file prefix used when processing a still image.",
    )
    args = parser.parse_args()

    try:
        if args.camera == "webcam":
            run_webcam_flow(args.index)
        elif args.camera == "ximea":
            run_ximea_flow()
        else:
            process_image_file(args.input, args.output_prefix)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
