import os
import glob
import cv2 
from ximea import xiapi


def capture_ximea(save_dir=None, prefix='capture', preview=False):
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_exposure(50000)
    cam.set_param("imgdataformat","XI_RGB32")
    cam.set_param("auto_wb",1)
    img = xiapi.Image()
    cam.start_acquisition()

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
        cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
        while True:
            cam.get_image(img)
            image = img.get_image_data_numpy()
            cv2.imshow('Preview', image)
            k = cv2.waitKey(30) & 0xFF
            if k == 32:  # SPACE 
                cam.get_image(img)
                image = img.get_image_data_numpy()
                fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
                cv2.imwrite(fname, image)
                frames.append(image)
                save_idx += 1
                print(f"Saved {len(frames)} images to {save_dir}")
            elif k == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
        cv2.imwrite(fname, image)
        frames.append(image)
        save_idx += 1

    cam.stop_acquisition()
    cam.close_device()

    return frames


def capture_webcam(save_dir=None, prefix='capture', preview=False):
    cap = cv2.VideoCapture(0)
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
        cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow('Preview', frame)
            k = cv2.waitKey(30) & 0xFF
            if k == 32:  # SPACE 
                fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
                cv2.imwrite(fname, frame)
                frames.append(frame)
                save_idx += 1
                print(f"Saved {len(frames)} images to {save_dir}")
            elif k == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        ret, frame = cap.read()
        if ret:
            fname = os.path.join(save_dir, f"{prefix}_{save_idx:03d}.jpg")
            cv2.imwrite(fname, frame)
            frames.append(frame)

    cap.release()
    return frames