
#adel and mariam from yohan projecct
import glob
import cv2
import numpy as np

from Plot.plot import show_image, show_plot
from Utils.io_utils import downscale_image, downscale_k, get_images
from config import (
    is_down_scale_images,
    checker_board_path,
    un_calibrated_path,
    pattern_size
)
def calibrate_camera(show_corners):
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane (pixels)

    # Criteria for sub-pixel accuracy in corner detection
    # This is CRUCIAL for good calibration!
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    h_cal, w_cal = 0, 0 # Initialize dimensions for calibration images

    # Load paths to calibration images
    calibration_image_paths = sorted(glob.glob(checker_board_path))

    for i, fname in enumerate(calibration_image_paths):
        img_cal = cv2.imread(fname)

        gray_cal = cv2.cvtColor(img_cal, cv2.COLOR_BGR2GRAY)
        h_cal, w_cal = gray_cal.shape[:2] # Get dimensions from current image

        ret, corners = cv2.findChessboardCorners(gray_cal, pattern_size, None)

        if ret:
            # Refine corner locations to sub-pixel accuracy
            cv2.cornerSubPix(gray_cal, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners)

            if i < 5 and show_corners:
                # Draw corners on a copy for display, avoiding modification of original img_cal
                img_display = img_cal.copy()
                cv2.drawChessboardCorners(img_display, pattern_size, corners, ret)
                img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                show_plot(img_rgb, f"Corners (Cal Image {i+1})")
        else:
            print(f"Info: Could not find chessboard corners in {fname}. Skipping this image for calibration.")
       
    # Perform camera calibration
    _, K, distCoeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w_cal, h_cal), None, None)
    
    print("\n--- Camera Calibration Results ---")
    print("Distortion Coefficients (distCoeffs):\n", distCoeffs) # IMPORTANT: Check this output!

    if is_down_scale_images:
        K = downscale_k(K.copy()) # Pass a copy to downscale_k to avoid modifying the original K in place accidentally if it's used elsewhere for logging
    
    # Load images for the SFM pipeline (assuming get_images returns actual image NumPy arrays)
    sfm_image_list = get_images(un_calibrated_path, '.png')
    
    return K, sfm_image_list, None
