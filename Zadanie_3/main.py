import numpy as np
import cv2 as cv


def load_image(path):
    try:
        return cv.imread(path)
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None

def save_image(path, image):
    try:
        cv.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")

def rbga_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        return image[:, :, :3]
    else:
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

def global_thresholding(image: np.ndarray, threshold: int) -> np.ndarray:
    
    grayscale = rgb_to_grayscale(image)
    return np.where(grayscale >= threshold, 255, 0).astype(np.uint8)

# def otsu_thresholding(image: np.ndarray, threshold: int) -> np.ndarray:

#     grayscale = rgb_to_grayscale(image)

#     histogram, _ = np.histogram(grayscale.flatten(), bins=256, range=(0, 255))
#     normized_histogram = histogram / histogram.sum()

#     if normized_histogram.any() < 0:
#         raise ValueError("Histogram contains negative values, which is not valid.")
    
#     omega_0 = np.sum(normized_histogram[:threshold])
#     omega_1 = np.sum(normized_histogram[threshold:])

#     assert omega_0 >= 0, "omega_0 must be non-negative"
#     assert omega_1 >= 0, "omega_1 must be non-negative"
#     assert omega_0 + omega_1 == 1, "The sum of omega_0 and omega_1 must equal 1"

#     mu_0 = [normized_histogram[:threshold][i] * i for i in range(threshold)]
#     mu_1 = [normized_histogram[threshold:][i] * i for i in range(threshold, 256)]

#     sigma_b_squared = omega_0 * omega_1 * (mu_0 - mu_1) ** 2

#     threshold_signed = np.argmax(sigma_b_squared)

#     return np.where(grayscale >= threshold_signed, 255, 0).astype(np.uint8)

def otsu_thresholding(image: np.ndarray, threshold: int = 0) -> np.ndarray:
    grayscale = rgb_to_grayscale(image).astype(np.uint8)

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
    return np.where(grayscale >= threshold_signed, 255, 0).astype(np.uint8)

if __name__ == "__main__":
    input_path = "glupik.jpg"
    output_path = "glupik_otsu.png"
    threshold_value = 128

    image = load_image(input_path)
    if image is not None:
        rgb_image = rbga_to_rgb(image)
        binary_image = otsu_thresholding(rgb_image, threshold_value)
        print(binary_image.shape)
        save_image(output_path, binary_image)



