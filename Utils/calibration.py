
#adel and mariam from yohan projecct
import glob
import cv2
import numpy as np

from Plot.plot import show_plot
from Utils.io_utils import downscale_k, get_images
from config import (
    is_down_scale_images,
    checker_board_path,
    un_calibrated_path,
    pattern_size
)

def calibrate_camera(show_corners):
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    gray = None
    h, w = None, None

    images = sorted(glob.glob(checker_board_path))

    for i, fname in enumerate(images):
        img = cv2.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if i < 5 and show_corners:
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                show_plot(img_rgb,f" Corners (Image {i+1})")
    _, K, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    
    if is_down_scale_images:
        K = downscale_k(K)
    
    # Load image files into the image list
    image_list = get_images(un_calibrated_path,'.png')
    
    return K,image_list
