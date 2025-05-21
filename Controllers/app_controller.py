import cv2
import numpy as np
from Utils.calibration import calibrate_camera
from Utils.common_points import com_points
from Utils.io_utils import downscale_image, downscale_k, get_images
from Utils.matching import features_matching
from Utils.optimization import bundle_adjustment
from Utils.triangulation import pnp_rasnac, reprojection_error, select_image_pair, triangulation
from Plot.plot import  show_results
from config import (
    is_down_scale_images,
    calibrated_path
)

from tqdm import tqdm


def init():
    # Load camera intrinsic matrix from file (K.txt)
    with open(calibrated_path + '/K.txt') as file:
        K = np.array(list((map(lambda x: list(map(lambda x: float(x), x.strip().split(' '))), file.read().split('\n')))))
    
    image_list = get_images(calibrated_path,'.jpg')
    

    if is_down_scale_images:
        # Downscale intrinsic matrix for image resizing
        K = downscale_k(K)
    
    return K, image_list

    
#adel
def get_ui_values(ba_var,data_var,matches_var,corner_var):
    return ba_var.get() == "Yes", data_var.get() == "Calibrate", matches_var.get() == "Yes",corner_var.get() == "Yes"

#adel and mariam
def start_sfm_process(ba_var,data_var,matches_var,corner_var):
    enable_bundle_adjustment,is_calibrate,show_matches,show_corners = get_ui_values(ba_var,data_var,matches_var,corner_var)
    if is_calibrate:
        K,image_list,dist_coeff= calibrate_camera(show_corners)
    else:
        K,image_list =init()
        dist_coeff=None
    
    print(f"\nCamera Matrix (K):\n{K}")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # First camera pose: identity
    first_transform_mat = np.hstack((np.eye(3), np.zeros((3, 1))))
    first_pose = K @ first_transform_mat
    total_points = np.zeros((1, 3))
    total_colors = np.zeros((1, 3))
    
    #after Zaar code review
    _, second_image, first_feature, second_feature,E = select_image_pair(image_list, K, show_matches)


    print(f"\nFundamental Matrix with RANSAC:\n{E}")
    #Fixes after Zaar code review
    # Normalize matched points
    first_norm = cv2.undistortPoints(np.expand_dims(first_feature, axis=1), K, dist_coeff)
    second_norm = cv2.undistortPoints(np.expand_dims(second_feature, axis=1), K, dist_coeff)

    # Decompose E into possible rotations and translations
    R1, R2, t = cv2.decomposeEssentialMat(E)
    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    best_pose = None
    max_positive = 0
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))

    for R, t in poses:
        P1 = np.hstack((R, t))
        pts_4d = cv2.triangulatePoints(K @ P0, K @ P1, first_norm, second_norm)
        pts_3d = pts_4d[:3] / pts_4d[3]

        z_cam0 = pts_3d[2]
        z_cam1 = (R[2] @ pts_3d) + t[2]
        num_positive = np.sum((z_cam0 > 0) & (z_cam1 > 0))

        if num_positive > max_positive:
            max_positive = num_positive
            best_pose = (R, t)

    if best_pose is None:
        raise RuntimeError("Could not determine correct pose.")

    rot_matrix, tran_matrix = best_pose
    print(f"\nRotation:\n{rot_matrix}")
    print(f"\nTranslation Matrix:\n{tran_matrix}")
    second_transform_mat = np.hstack((rot_matrix, tran_matrix))
    
    pose_4x4_first = np.eye(4)
    pose_4x4_first[:3, :4] = first_transform_mat
    pose_4x4_second = np.eye(4)
    pose_4x4_second[:3, :4] = second_transform_mat

    camera_poses = []

    camera_poses.append(pose_4x4_first)
    camera_poses.append(pose_4x4_second)

    second_pose = K @ second_transform_mat

    # Triangulate 3D points
    first_feature, second_feature, points_3d = triangulation(first_pose, second_pose, first_feature, second_feature)

    # Compute reprojection error for the triangulated points
    error, points_3d = reprojection_error(points_3d, second_feature, second_transform_mat, K, homogenity = 1,dist_coeff=dist_coeff)

    print("first rp error: ", error)
    ##---
    ##---Incremental Expansion: Add new cameras via PnP, triangulate points, and refine via bundle adjustment.

    #Initial pose estimation using PnP algorithm and first set of features
    _, _, second_feature, points_3d, _ = pnp_rasnac(points_3d, second_feature, K,  dist_coeff if is_calibrate else np.zeros((5, 1), dtype=np.float32), first_feature, initial=1)

    # Bundle adjustment preparation
    total_images = len(image_list) - 2
    pose_array = np.hstack((K.ravel(), first_pose.ravel(), second_pose.ravel()))
    threshold = 0.5

    errors=[]
    all_points_3d = [points_3d]  # Store all triangulated 3D points
    all_image_features = [first_feature, second_feature] # Store corresponding image features

    # Loop through each subsequent image for incremental camera expansion
    for i in tqdm(range(total_images)):
        # Load and downscale current image for processing
        current_image = downscale_image(image_list[i + 2])

        # Feature matching between previous and current image
        features_prev, features_curr, _ = features_matching(second_image, current_image, show_matches, K)
        all_image_features.append(features_curr)

        # If not the first iteration, perform triangulation for 3D points
        if i != 0:
            first_feature, second_feature, points_3d = triangulation(first_pose, second_pose, first_feature, second_feature)
            second_feature = second_feature.T
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
            points_3d = points_3d[:, 0, :]
            all_points_3d.append(points_3d)

        # Find common points between current and previous images
        cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = com_points(second_feature, features_prev, features_curr, is_pre_calibrated=is_calibrate)

        cm_points_2 = features_curr[cm_points_1]
        cm_points_cur = features_prev[cm_points_1]

        # Estimate the pose of the new camera using PnP
        rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = pnp_rasnac(points_3d[cm_points_0], cm_points_2, K,  dist_coeff if is_calibrate else np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial=0)
        second_transform_mat = np.hstack((rot_matrix, tran_matrix))  # Combine rotation and translation into a single matrix
        pose_2 = np.matmul(K, second_transform_mat)  # Camera pose in world coordinates

        # Calculate reprojection error for the newly estimated 3D points
        error, points_3d = reprojection_error(points_3d, cm_points_2, second_transform_mat, K, homogenity=0,dist_coeff=dist_coeff)

        # Perform triangulation to estimate 3D points from the new camera pose
        cm_mask_0, cm_mask_1, points_3d = triangulation(second_pose, pose_2, cm_mask_0, cm_mask_1)
        error, points_3d = reprojection_error(points_3d, cm_mask_1, second_transform_mat, K, homogenity=1,dist_coeff=dist_coeff)
        print("rp error: ", error)
        all_points_3d.append(points_3d)

        # Add the new pose to the pose array
        pose_array = np.hstack((pose_array, pose_2.ravel()))
        # Store extrinsic pose as 4x4 for Open3D visualization
        Rt = np.hstack((rot_matrix, tran_matrix))  # 3x4
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :4] = Rt
        camera_poses.append(pose_4x4)  # Store for visualization

        current_3d_points = None
        current_colors = None

        # Perform bundle adjustment to refine 3D points and camera poses, if enabled
        if enable_bundle_adjustment:
            points_3d, cm_mask_1, second_transform_mat,K = bundle_adjustment(points_3d, cm_mask_1, second_transform_mat, K, threshold,dist_coeff,is_calibrate)
            pose_2 = np.matmul(K, second_transform_mat)  # Update the refined pose
            error, points_3d = reprojection_error(points_3d, cm_mask_1, second_transform_mat, K, homogenity=0,dist_coeff=dist_coeff)
            print("Bundle error: ", error)
            current_3d_points = points_3d
            points_left = np.array(cm_mask_1, dtype=np.int32)
            height, width = current_image.shape[:2]
            color_vector = np.array([
                current_image[l[1], l[0]] 
                for l in points_left 
                if 0 <= l[0] < width and 0 <= l[1] < height
            ])
            color_vector = cv2.cvtColor(color_vector.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
            current_colors = color_vector

        else:
            # Without bundle adjustment, just accumulate the points
            current_3d_points = points_3d[:, 0, :]
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([current_image[l[1], l[0]] for l in points_left.T])
            color_vector = cv2.cvtColor(color_vector.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
            current_colors = color_vector

        if current_3d_points is not None and current_colors is not None:
            if total_points.shape[0] == 1: # Initialize
                total_points = current_3d_points
                total_colors = current_colors
            else:
                total_points = np.vstack((total_points, current_3d_points))
                total_colors = np.vstack((total_colors, current_colors))

        # Prepare for the next iteration: update pose and feature points
        first_transform_mat = np.copy(second_transform_mat)
        first_pose = np.copy(second_pose)
        #to show latererrors
        errors.append(error)
        # Update images and feature sets for the next iteration
        second_image = np.copy(current_image)
        first_feature = np.copy(features_prev)
        second_feature = np.copy(features_curr)
        second_pose = np.copy(pose_2)

        # Display the current image and check for a break condition
        cv2.imshow(f"Image {i}", current_image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # Close all OpenCV windows after processing is complete
    cv2.destroyAllWindows()

    show_results(errors, camera_poses, total_points[1:], total_colors[1:]) # Exclude initial zero arrays
