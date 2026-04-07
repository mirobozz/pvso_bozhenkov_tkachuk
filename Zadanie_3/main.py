import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

try:
    from ximea import xiapi
except ImportError:
    xiapi = None


WINDOW_NAME = "VideoFlow"
CONTROL_WINDOW_NAME = "Threshold Controls"
DEFAULT_INPUT_PATH = str(Path(__file__).resolve().with_name("glupik.jpg"))
DISPLAY_TILE_SIZE = (480, 360)
GLOBAL_THRESHOLD = 128
ADAPTIVE_WINDOW_SIZE = 31
ADAPTIVE_T_PERCENT = 7.5
OPENCV_ADAPTIVE_C = 7.5
MAX_GLOBAL_THRESHOLD = 255
MAX_ADAPTIVE_WINDOW_SIZE = 255
MAX_ADAPTIVE_T_PERCENT_TENTHS = 1000
MAX_OPENCV_ADAPTIVE_C_TENTHS = 1000
DEFAULT_BENCHMARK_ITERATIONS = 200

MODE_DEFAULT = "default"
MODE_COMPARE = "compare"
MODE_CHOICES = [MODE_DEFAULT, MODE_COMPARE]

METHOD_GLOBAL = "global"
METHOD_OTSU = "otsu"
METHOD_ADAPTIVE = "adaptive"
METHOD_CHOICES = [METHOD_GLOBAL, METHOD_OTSU, METHOD_ADAPTIVE]
METHOD_TITLES = {
    METHOD_GLOBAL: "Global",
    METHOD_OTSU: "Otsu",
    METHOD_ADAPTIVE: "Adaptive",
}

GLOBAL_THRESHOLD_TRACKBAR = "Global threshold"
ADAPTIVE_WINDOW_TRACKBAR = "Adaptive window"
ADAPTIVE_T_TRACKBAR = "Custom T x0.1%"
OPENCV_ADAPTIVE_C_TRACKBAR = "OpenCV C x0.1"


@dataclass(frozen=True)
class ThresholdSettings:
    global_threshold: int = GLOBAL_THRESHOLD
    adaptive_window_size: int = ADAPTIVE_WINDOW_SIZE
    adaptive_t_percent: float = ADAPTIVE_T_PERCENT
    opencv_adaptive_c: float = OPENCV_ADAPTIVE_C


@dataclass(frozen=True)
class ComparisonResult:
    custom_binary: np.ndarray
    opencv_binary: np.ndarray
    custom_label: str
    opencv_label: str


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


def opencv_global_thresholding_gray(grayscale: np.ndarray, threshold: int) -> np.ndarray:
    _, binary = cv.threshold(grayscale.astype(np.uint8), threshold, 255, cv.THRESH_BINARY)
    return binary.astype(np.uint8)


def opencv_otsu_thresholding_gray(grayscale: np.ndarray) -> np.ndarray:
    _, binary = cv.threshold(grayscale.astype(np.uint8), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary.astype(np.uint8)


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


def normalize_adaptive_window_size(window_size: int) -> int:
    window_size = max(1, min(MAX_ADAPTIVE_WINDOW_SIZE, window_size))
    if window_size % 2 == 0:
        window_size += 1 if window_size < MAX_ADAPTIVE_WINDOW_SIZE else -1
    return window_size


def opencv_adaptive_mean_thresholding_gray(
    grayscale: np.ndarray,
    window_size: int,
    c_value: float,
) -> np.ndarray:
    return cv.adaptiveThreshold(
        grayscale.astype(np.uint8),
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        normalize_adaptive_window_size(window_size),
        c_value,
    ).astype(np.uint8)


def on_trackbar_change(_value: int):
    pass


def method_has_controls(method_name: str) -> bool:
    return method_name in {METHOD_GLOBAL, METHOD_ADAPTIVE}


def mode_has_controls(mode_name: str, method_name: str) -> bool:
    if mode_name == MODE_DEFAULT:
        return True
    if mode_name == MODE_COMPARE:
        return method_has_controls(method_name)
    raise ValueError(f"Unsupported mode: {mode_name}")


def setup_threshold_controls(mode_name: str, method_name: str):
    if not mode_has_controls(mode_name, method_name):
        return

    cv.namedWindow(CONTROL_WINDOW_NAME, cv.WINDOW_NORMAL)
    control_height = 200
    if mode_name == MODE_COMPARE and method_name == METHOD_GLOBAL:
        control_height = 110
    cv.resizeWindow(CONTROL_WINDOW_NAME, 520, control_height)

    if mode_name == MODE_DEFAULT:
        cv.createTrackbar(
            GLOBAL_THRESHOLD_TRACKBAR,
            CONTROL_WINDOW_NAME,
            GLOBAL_THRESHOLD,
            MAX_GLOBAL_THRESHOLD,
            on_trackbar_change,
        )
        cv.createTrackbar(
            ADAPTIVE_WINDOW_TRACKBAR,
            CONTROL_WINDOW_NAME,
            ADAPTIVE_WINDOW_SIZE,
            MAX_ADAPTIVE_WINDOW_SIZE,
            on_trackbar_change,
        )
        cv.createTrackbar(
            ADAPTIVE_T_TRACKBAR,
            CONTROL_WINDOW_NAME,
            int(ADAPTIVE_T_PERCENT * 10),
            MAX_ADAPTIVE_T_PERCENT_TENTHS,
            on_trackbar_change,
        )
        return

    if method_name == METHOD_GLOBAL:
        cv.createTrackbar(
            GLOBAL_THRESHOLD_TRACKBAR,
            CONTROL_WINDOW_NAME,
            GLOBAL_THRESHOLD,
            MAX_GLOBAL_THRESHOLD,
            on_trackbar_change,
        )
        return

    cv.createTrackbar(
        ADAPTIVE_WINDOW_TRACKBAR,
        CONTROL_WINDOW_NAME,
        ADAPTIVE_WINDOW_SIZE,
        MAX_ADAPTIVE_WINDOW_SIZE,
        on_trackbar_change,
    )
    cv.createTrackbar(
        ADAPTIVE_T_TRACKBAR,
        CONTROL_WINDOW_NAME,
        int(ADAPTIVE_T_PERCENT * 10),
        MAX_ADAPTIVE_T_PERCENT_TENTHS,
        on_trackbar_change,
    )
    cv.createTrackbar(
        OPENCV_ADAPTIVE_C_TRACKBAR,
        CONTROL_WINDOW_NAME,
        int(OPENCV_ADAPTIVE_C * 10),
        MAX_OPENCV_ADAPTIVE_C_TENTHS,
        on_trackbar_change,
    )


def get_threshold_settings(mode_name: str, method_name: str) -> ThresholdSettings:
    defaults = ThresholdSettings()

    if mode_name == MODE_DEFAULT:
        global_threshold = cv.getTrackbarPos(GLOBAL_THRESHOLD_TRACKBAR, CONTROL_WINDOW_NAME)
        adaptive_window_size = cv.getTrackbarPos(ADAPTIVE_WINDOW_TRACKBAR, CONTROL_WINDOW_NAME)
        adaptive_window_size = normalize_adaptive_window_size(adaptive_window_size)
        cv.setTrackbarPos(ADAPTIVE_WINDOW_TRACKBAR, CONTROL_WINDOW_NAME, adaptive_window_size)

        return ThresholdSettings(
            global_threshold=global_threshold,
            adaptive_window_size=adaptive_window_size,
            adaptive_t_percent=cv.getTrackbarPos(ADAPTIVE_T_TRACKBAR, CONTROL_WINDOW_NAME) / 10.0,
            opencv_adaptive_c=defaults.opencv_adaptive_c,
        )

    if mode_name != MODE_COMPARE:
        raise ValueError(f"Unsupported mode: {mode_name}")

    if method_name == METHOD_GLOBAL:
        return ThresholdSettings(
            global_threshold=cv.getTrackbarPos(GLOBAL_THRESHOLD_TRACKBAR, CONTROL_WINDOW_NAME),
            adaptive_window_size=defaults.adaptive_window_size,
            adaptive_t_percent=defaults.adaptive_t_percent,
            opencv_adaptive_c=defaults.opencv_adaptive_c,
        )

    if method_name == METHOD_ADAPTIVE:
        adaptive_window_size = cv.getTrackbarPos(ADAPTIVE_WINDOW_TRACKBAR, CONTROL_WINDOW_NAME)
        adaptive_window_size = normalize_adaptive_window_size(adaptive_window_size)
        cv.setTrackbarPos(ADAPTIVE_WINDOW_TRACKBAR, CONTROL_WINDOW_NAME, adaptive_window_size)

        return ThresholdSettings(
            global_threshold=defaults.global_threshold,
            adaptive_window_size=adaptive_window_size,
            adaptive_t_percent=cv.getTrackbarPos(ADAPTIVE_T_TRACKBAR, CONTROL_WINDOW_NAME) / 10.0,
            opencv_adaptive_c=cv.getTrackbarPos(OPENCV_ADAPTIVE_C_TRACKBAR, CONTROL_WINDOW_NAME) / 10.0,
        )

    return defaults


def setup_display_windows(mode_name: str, method_name: str):
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, DISPLAY_TILE_SIZE[0] * 2, DISPLAY_TILE_SIZE[1] * 2)
    setup_threshold_controls(mode_name, method_name)


def is_window_closed(window_name: str) -> bool:
    try:
        return cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1
    except cv.error:
        return True


def resize_for_tile(image: np.ndarray) -> np.ndarray:
    return cv.resize(image, DISPLAY_TILE_SIZE, interpolation=cv.INTER_AREA)


def threshold_to_bgr(threshold_image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(threshold_image, cv.COLOR_GRAY2BGR)


def annotate_tile(image: np.ndarray, label: str) -> np.ndarray:
    annotated = image.copy()
    for idx, line in enumerate(label.splitlines()):
        cv.putText(
            annotated,
            line,
            (12, 30 + idx * 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
    return annotated


def run_custom_method_gray(
    grayscale: np.ndarray,
    settings: ThresholdSettings,
    method_name: str,
) -> tuple[np.ndarray, str]:
    if method_name == METHOD_GLOBAL:
        return (
            global_thresholding_gray(grayscale, settings.global_threshold),
            f"Custom Global\nT={settings.global_threshold}",
        )
    if method_name == METHOD_OTSU:
        return otsu_thresholding_gray(grayscale), "Custom Otsu"
    if method_name == METHOD_ADAPTIVE:
        return (
            adaptive_mean_thresholding_percent_gray(
                grayscale,
                settings.adaptive_window_size,
                settings.adaptive_t_percent,
            ),
            f"Custom Adaptive\nW={settings.adaptive_window_size} T={settings.adaptive_t_percent:.1f}%",
        )
    raise ValueError(f"Unsupported method: {method_name}")


def run_opencv_method_gray(
    grayscale: np.ndarray,
    settings: ThresholdSettings,
    method_name: str,
) -> tuple[np.ndarray, str]:
    if method_name == METHOD_GLOBAL:
        return (
            opencv_global_thresholding_gray(grayscale, settings.global_threshold),
            f"OpenCV Global\nT={settings.global_threshold}",
        )
    if method_name == METHOD_OTSU:
        return opencv_otsu_thresholding_gray(grayscale), "OpenCV Otsu"
    if method_name == METHOD_ADAPTIVE:
        return (
            opencv_adaptive_mean_thresholding_gray(
                grayscale,
                settings.adaptive_window_size,
                settings.opencv_adaptive_c,
            ),
            f"OpenCV Adaptive\nW={settings.adaptive_window_size} C={settings.opencv_adaptive_c:.1f}",
        )
    raise ValueError(f"Unsupported method: {method_name}")


def build_comparison_result(
    grayscale: np.ndarray,
    settings: ThresholdSettings,
    method_name: str,
) -> ComparisonResult:
    custom_binary, custom_label = run_custom_method_gray(grayscale, settings, method_name)
    opencv_binary, opencv_label = run_opencv_method_gray(grayscale, settings, method_name)
    return ComparisonResult(
        custom_binary=custom_binary,
        opencv_binary=opencv_binary,
        custom_label=custom_label,
        opencv_label=opencv_label,
    )


def build_default_panel(
    frame: np.ndarray,
    settings: ThresholdSettings,
) -> np.ndarray:
    frame = resize_for_tile(frame)
    grayscale = rgb_to_grayscale(frame)

    global_binary = global_thresholding_gray(grayscale, settings.global_threshold)
    otsu_binary = otsu_thresholding_gray(grayscale)
    adaptive_binary = adaptive_mean_thresholding_percent_gray(
        grayscale,
        settings.adaptive_window_size,
        settings.adaptive_t_percent,
    )

    original_tile = annotate_tile(frame, "Original")
    global_tile = annotate_tile(
        threshold_to_bgr(global_binary),
        f"Global\nT={settings.global_threshold}",
    )
    otsu_tile = annotate_tile(threshold_to_bgr(otsu_binary), "Otsu")
    adaptive_tile = annotate_tile(
        threshold_to_bgr(adaptive_binary),
        f"Adaptive\nW={settings.adaptive_window_size} T={settings.adaptive_t_percent:.1f}%",
    )

    top_row = cv.hconcat([original_tile, global_tile])
    bottom_row = cv.hconcat([otsu_tile, adaptive_tile])
    return cv.vconcat([top_row, bottom_row])


def build_comparison_panel(
    frame: np.ndarray,
    settings: ThresholdSettings,
    method_name: str,
) -> np.ndarray:
    frame = resize_for_tile(frame)
    grayscale = rgb_to_grayscale(frame)
    comparison = build_comparison_result(grayscale, settings, method_name)
    diff_binary = cv.absdiff(comparison.custom_binary, comparison.opencv_binary)
    diff_pixels = int(cv.countNonZero(diff_binary))

    original_tile = annotate_tile(frame, "Original")
    global_tile = annotate_tile(
        threshold_to_bgr(comparison.custom_binary),
        comparison.custom_label,
    )
    adaptive_tile = annotate_tile(
        threshold_to_bgr(comparison.opencv_binary),
        comparison.opencv_label,
    )
    diff_tile = annotate_tile(threshold_to_bgr(diff_binary), f"Abs Diff\n{diff_pixels} px")

    top_row = cv.hconcat([original_tile, global_tile])
    bottom_row = cv.hconcat([adaptive_tile, diff_tile])
    return cv.vconcat([top_row, bottom_row])


def build_video_panel(
    frame: np.ndarray,
    settings: ThresholdSettings,
    mode_name: str,
    method_name: str,
) -> np.ndarray:
    if mode_name == MODE_DEFAULT:
        return build_default_panel(frame, settings)
    if mode_name == MODE_COMPARE:
        return build_comparison_panel(frame, settings, method_name)
    raise ValueError(f"Unsupported mode: {mode_name}")


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


def save_default_outputs(
    image: np.ndarray,
    output_prefix: str,
    settings: ThresholdSettings,
):
    global_image = global_thresholding(image, settings.global_threshold)
    otsu_image = otsu_thresholding(image)
    adaptive_image = adaptive_mean_thresholding_percent(
        image,
        window_size=settings.adaptive_window_size,
        t_percent=settings.adaptive_t_percent,
    )
    panel = build_default_panel(image, settings)

    save_image(f"{output_prefix}_global.png", global_image)
    save_image(f"{output_prefix}_otsu.png", otsu_image)
    save_image(f"{output_prefix}_adaptive.png", adaptive_image)
    save_image(f"{output_prefix}_panel.png", panel)
    print(f"Saved default outputs with prefix '{output_prefix}'.")


def save_comparison_outputs(
    image: np.ndarray,
    output_prefix: str,
    settings: ThresholdSettings,
    method_name: str,
):
    grayscale = rgb_to_grayscale(image)
    comparison = build_comparison_result(grayscale, settings, method_name)
    diff_image = cv.absdiff(comparison.custom_binary, comparison.opencv_binary)
    panel = build_comparison_panel(image, settings, method_name)

    save_image(f"{output_prefix}_{method_name}_custom.png", comparison.custom_binary)
    save_image(f"{output_prefix}_{method_name}_opencv.png", comparison.opencv_binary)
    save_image(f"{output_prefix}_{method_name}_diff.png", diff_image)
    save_image(f"{output_prefix}_{method_name}_panel.png", panel)
    print(f"Saved {METHOD_TITLES[method_name]} comparison outputs with prefix '{output_prefix}'.")


def save_threshold_outputs(
    image: np.ndarray,
    output_prefix: str,
    settings: ThresholdSettings,
    mode_name: str,
    method_name: str,
):
    if mode_name == MODE_DEFAULT:
        save_default_outputs(image, output_prefix, settings)
        return
    if mode_name == MODE_COMPARE:
        save_comparison_outputs(image, output_prefix, settings, method_name)
        return
    raise ValueError(f"Unsupported mode: {mode_name}")


def build_runtime_message(source_name: str, mode_name: str, method_name: str) -> str:
    if mode_name == MODE_DEFAULT:
        return (
            f"{source_name} default thresholding mode running. "
            "Adjust sliders in 'Threshold Controls'. Press 'q' to quit."
        )

    control_message = (
        "Adjust sliders in 'Threshold Controls'. "
        if mode_has_controls(mode_name, method_name)
        else "No sliders are needed for this method. "
    )
    return (
        f"{source_name} {METHOD_TITLES[method_name]} comparison running. "
        f"{control_message}Press 'q' to quit."
    )


def control_window_closed(mode_name: str, method_name: str) -> bool:
    return mode_has_controls(mode_name, method_name) and is_window_closed(CONTROL_WINDOW_NAME)


def run_webcam_flow(device: int, mode_name: str, method_name: str):
    cap = cv.VideoCapture(device, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {device}")

    setup_display_windows(mode_name, method_name)
    print(build_runtime_message("Webcam", mode_name, method_name))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam.")

            frame = cv.flip(frame, 1)
            settings = get_threshold_settings(mode_name, method_name)
            panel = build_video_panel(frame, settings, mode_name, method_name)
            cv.imshow(WINDOW_NAME, panel)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or is_window_closed(WINDOW_NAME) or control_window_closed(mode_name, method_name):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def run_ximea_flow(mode_name: str, method_name: str):
    cam, img = setup_ximea_camera()

    setup_display_windows(mode_name, method_name)
    print(build_runtime_message("Ximea", mode_name, method_name))

    try:
        while True:
            frame = read_ximea_frame(cam, img)
            settings = get_threshold_settings(mode_name, method_name)
            panel = build_video_panel(frame, settings, mode_name, method_name)
            cv.imshow(WINDOW_NAME, panel)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or is_window_closed(WINDOW_NAME) or control_window_closed(mode_name, method_name):
                break
    finally:
        cam.stop_acquisition()
        cam.close_device()
        cv.destroyAllWindows()


def process_image_file(input_path: str, output_prefix: str, mode_name: str, method_name: str):
    image = load_image(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    image = rbga_to_rgb(image)
    setup_display_windows(mode_name, method_name)
    if mode_name == MODE_DEFAULT:
        print("Image default thresholding mode running. Adjust sliders in 'Threshold Controls'. Press 's' to save or 'q' to save and quit.")
    else:
        control_message = (
            "Adjust sliders in 'Threshold Controls'. "
            if mode_has_controls(mode_name, method_name)
            else "No sliders are needed for this method. "
        )
        print(f"Image {METHOD_TITLES[method_name]} comparison running. {control_message}Press 's' to save or 'q' to save and quit.")

    last_settings = ThresholdSettings()

    try:
        while True:
            last_settings = get_threshold_settings(mode_name, method_name)
            panel = build_video_panel(image, last_settings, mode_name, method_name)
            cv.imshow(WINDOW_NAME, panel)

            key = cv.waitKey(30) & 0xFF
            if key == ord("s"):
                save_threshold_outputs(image, output_prefix, last_settings, mode_name, method_name)
            if key == ord("q") or is_window_closed(WINDOW_NAME) or control_window_closed(mode_name, method_name):
                save_threshold_outputs(image, output_prefix, last_settings, mode_name, method_name)
                break
    finally:
        cv.destroyAllWindows()


def benchmark_operation(operation, iterations: int) -> tuple[float, float]:
    for _ in range(5):
        operation()

    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return float("inf"), 0.0
    return iterations / elapsed, (elapsed / iterations) * 1000.0


def run_benchmark(input_path: str, method_name: str, iterations: int):
    if iterations <= 0:
        raise ValueError("Benchmark iterations must be a positive integer.")

    image = load_image(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    image = rbga_to_rgb(image)
    grayscale = rgb_to_grayscale(image)
    settings = ThresholdSettings()

    custom_fps, custom_ms = benchmark_operation(
        lambda: run_custom_method_gray(grayscale, settings, method_name)[0],
        iterations,
    )
    opencv_fps, opencv_ms = benchmark_operation(
        lambda: run_opencv_method_gray(grayscale, settings, method_name)[0],
        iterations,
    )

    comparison = build_comparison_result(grayscale, settings, method_name)
    diff_pixels = int(cv.countNonZero(cv.absdiff(comparison.custom_binary, comparison.opencv_binary)))

    print(f"Benchmark method: {METHOD_TITLES[method_name]}")
    print(f"Image size: {grayscale.shape[1]}x{grayscale.shape[0]}")
    print(f"Iterations: {iterations}")
    print(f"Custom FPS: {custom_fps:.2f} ({custom_ms:.3f} ms/frame)")
    print(f"OpenCV FPS: {opencv_fps:.2f} ({opencv_ms:.3f} ms/frame)")
    print(f"Pixel differences: {diff_pixels}")


def main():
    parser = argparse.ArgumentParser(description="Thresholding app with default and comparison modes.")
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
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=MODE_DEFAULT,
        help="Runtime mode: 'default' restores the original custom-only view, 'compare' shows custom vs OpenCV.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=METHOD_CHOICES,
        default=METHOD_GLOBAL,
        help="Thresholding method used in compare mode and benchmark mode.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run an FPS benchmark on --input for the selected method and exit.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=DEFAULT_BENCHMARK_ITERATIONS,
        help="Number of iterations used when --benchmark is enabled.",
    )
    args = parser.parse_args()

    try:
        if args.benchmark:
            run_benchmark(args.input, args.method, args.benchmark_iterations)
        elif args.camera == "webcam":
            run_webcam_flow(args.index, args.mode, args.method)
        elif args.camera == "ximea":
            run_ximea_flow(args.mode, args.method)
        else:
            process_image_file(args.input, args.output_prefix, args.mode, args.method)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
