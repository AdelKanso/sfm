
#adel
import cv2
import numpy as np

from Utils.io_utils import downscale_image
from Utils.matching import features_matching


def triangulation(first_2d, second_2d, first_proj_matrix, second_proj_matrix) -> tuple:
    pt_cloud = cv2.triangulatePoints(first_2d, second_2d, first_proj_matrix.T, second_proj_matrix.T)
    return first_proj_matrix.T, second_proj_matrix.T, (pt_cloud / pt_cloud[3])  # Normalize by the homogeneous coordinate

#adel
def pnp_rasnac(obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
    if initial == 1:
        obj_point = obj_point[:, 0, :]
        image_point = image_point.T
        rot_vector = rot_vector.T 
    
    # Solve PnP using RANSAC to estimate rotation and translation
    _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
    
    # Convert rotation vector to rotation matrix
    rot_m, _ = cv2.Rodrigues(rot_vector_calc)

    # Filter points based on inliers
    if inlier is not None:
        image_point = image_point[inlier[:, 0]]
        obj_point = obj_point[inlier[:, 0]]
        rot_vector = rot_vector[inlier[:, 0]]
    
    return rot_m, tran_vector, image_point, obj_point, rot_vector

#mariam
def reprojection_error(obj_points, image_points, transf_mat, K, homogenity) -> tuple:
    rot_m = transf_mat[:3, :3]
    tran_vector = transf_mat[:3, 3]
    rot_vector, _ = cv2.Rodrigues(rot_m)
    
    # Convert points to homogeneous
    if homogenity == 1:
        obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
    
    # Project object points onto image plane
    image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
    image_points_calc = np.float32(image_points_calc[:, 0, :])
    
    # Calculate the reprojection error
    total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
    return total_error / len(image_points_calc), obj_points

def select_image_pair(image_list, K, show_matches):
    """ Selects an image pair with sufficient baseline using SIFT matches. """
    min_parallax_threshold = 40  # Chosen after testing on data sets
    
    for i in range(len(image_list) - 1):
        img1 = downscale_image(image_list[i])
        img2 = downscale_image(image_list[i + 1])

        # Feature matching
        keypoints1, keypoints2, E = features_matching(img1, img2, show_matches, K)

        # Compute disparity (parallax)
        parallax = np.mean(np.linalg.norm(keypoints1 - keypoints2, axis=1))

        if parallax > min_parallax_threshold:
            print(f"Selected Image Pair: {i}, {i + 1} with parallax {parallax:.2f}")
            return img1, img2, keypoints1, keypoints2,E

    raise RuntimeError("No suitable image pair found with sufficient baseline.")
