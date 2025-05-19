import cv2
import os
import numpy as np
from scipy.optimize import least_squares
from plot import show_image,show_plot
import glob
from plot import traj_plot,sparse_save
import matplotlib.pyplot as plt
from tqdm import tqdm


# Downscale factor for image resizing and directory path for images
reduce_res_fac = 2.0
is_down_scale_images=True
calibrated_path = "Datasets/Calibrated"
checker_board_path="Datasets/Images/cal/*.jpeg"
un_calibrated_path="Datasets/Images"
#used for calibration checker board
pattern_size=(9, 7)


def get_images(path,extension):
    image_list = []
    for image in sorted(os.listdir(path)):
        if image[-4:].lower() == extension:
            image_list.append(path + '/' + image)
    return image_list

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
def downscale_k(K):
    K[0, 0] /= reduce_res_fac
    K[1, 1] /= reduce_res_fac
    K[0, 2] /= reduce_res_fac
    K[1, 2] /= reduce_res_fac
    return K
#adel
def downscale_image(image):
    # Downscale image using pyramid down-scaling to reduce resolution
    if is_down_scale_images:
        for _ in range(1, int(reduce_res_fac / 2) + 1):
            image = cv2.pyrDown(image)
    return image
#adel and mariam from yohan projecct
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

#mariam
def features_matching(first_img, second_img,show_matches,K) -> tuple:
    # Initialize SIFT feature detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect keypoints and compute descriptors for both images (converted to grayscale)
    first_key_points, first_descriptors = sift.detectAndCompute(cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY), None)
    second_key_points, second_descriptors = sift.detectAndCompute(cv2.cvtColor(second_img, cv2.COLOR_BGR2GRAY), None)

    # Use Brute-Force matcher to find the two best founded_matches for each descriptor
    bf = cv2.BFMatcher()
    founded_matches = bf.knnMatch(first_descriptors, second_descriptors, k=2)

    feature = []
    # Apply Lowe's ratio test to keep good founded_matches (to remove ambiguous ones)
    for m, n in founded_matches:
        if m.distance < 0.70* n.distance:
            feature.append(m)
            
    if show_matches:
        #show founded_matches
        matched_image = cv2.drawMatches(first_img, first_key_points, second_img, second_key_points, feature, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        show_image("Matches", matched_image)

    pts0=np.float32([first_key_points[m.queryIdx].pt for m in feature])
    pts1=np.float32([second_key_points[m.trainIdx].pt for m in feature])
    # Find Fundamental Matrix with RANSAC
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)

    #calculate essential matrix
    E = K.T @ F @ K
    # Return coordinates of matched keypoints
    # Select only inlier founded_matches
    return pts0[mask.ravel() == 1],pts1[mask.ravel() == 1],E
#adel
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

#mariam and adel
def opt_reprojection_error(obj_points) -> np.array:
    transf_mat = obj_points[0:12].reshape((3, 4))
    K = obj_points[12:21].reshape((3, 3))
    rest = int(len(obj_points[21:]) * 0.4)
    p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
    obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:]) / 3), 3))
    
    rot_m = transf_mat[:3, :3]
    tran_vector = transf_mat[:3, 3]
    rot_vector, _ = cv2.Rodrigues(rot_m)
    
    # Project object points and calculate the error
    image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
    image_points = image_points[:, 0, :]
    error = [(p[idx] - image_points[idx])**2 for idx in range(len(p))]
    return np.array(error).ravel() / len(p)
#mariam and adel
def bundle_adjustment(_3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
    opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
    opt_variables = np.hstack((opt_variables, opt.ravel()))
    opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

    # Perform least squares optimization to minimize reprojection error
    values_corrected = least_squares(opt_reprojection_error, opt_variables, gtol=r_error).x
    K = values_corrected[12:21].reshape((3, 3))
    rest = int(len(values_corrected[21:]) * 0.4)
    return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:]) / 3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest / 2))).T, values_corrected[0:12].reshape((3, 4))
#mariam
def com_points(first_points, second_points, third_points,is_pre_calibrated=True) -> tuple:
    # Find common points between the first and second sets by matching coordinates
    first_cm_points = []
    second_cm_points = []
    if is_pre_calibrated:
        for i in range(first_points.shape[0]):
            a = np.where(second_points == first_points[i, :])
            if a[0].size != 0:
                first_cm_points.append(i)
                second_cm_points.append(a[0][0])
    else:
        for i in range(first_points.shape[0]):
            matches = np.all(second_points == first_points[i], axis=1)
            if np.any(matches):
                first_cm_points.append(i)
                second_cm_points.append(np.where(matches)[0][0])

    # Mask out the matched points from the second and third sets
    first_mask_array = np.ma.array(second_points, mask=False)
    first_mask_array.mask[second_cm_points] = True
    first_mask_array = first_mask_array.compressed().reshape(-1, 2)

    second_mask_array = np.ma.array(third_points, mask=False)
    second_mask_array.mask[second_cm_points] = True
    second_mask_array = second_mask_array.compressed().reshape(-1, 2)

    return np.array(first_cm_points), np.array(second_cm_points), first_mask_array, second_mask_array

#mariam
def show_results(errors,camera_poses,total_points,total_colors):
    plt.title("Reprojection Errors")
    for i in range(len(errors)):
        plt.scatter(i, errors[i])
        
    plt.show()

    # Visualize the sparse 3D points and save
    sparse_save(total_points, total_colors,with_colors=False)
    # Show poses
    traj_plot(camera_poses)
    # Show Colorized and save
    sparse_save(total_points, total_colors,with_colors=True)
    
#adel
def get_ui_values(ba_var,data_var,matches_var,corner_var):
    return ba_var.get() == "Yes", data_var.get() == "Calibrate", matches_var.get() == "Yes",corner_var.get() == "Yes"

#adel
def select_image_pair(image_list, K, show_matches):
    """ Selects an image pair with sufficient baseline using SIFT matches. """
    min_parallax_threshold = 40  # Chosen after testing on data sets
    
    for i in range(len(image_list) - 1):
        img1 = downscale_image(cv2.imread(image_list[i]))
        img2 = downscale_image(cv2.imread(image_list[i + 1]))

        # Feature matching
        keypoints1, keypoints2, E = features_matching(img1, img2, show_matches, K)

        # Compute disparity (parallax)
        parallax = np.mean(np.linalg.norm(keypoints1 - keypoints2, axis=1))

        if parallax > min_parallax_threshold:
            print(f"Selected Image Pair: {i}, {i + 1} with parallax {parallax:.2f}")
            return img1, img2, keypoints1, keypoints2,E

    raise RuntimeError("No suitable image pair found with sufficient baseline.")

#adel and mariam
def start_sfm_process(ba_var, data_var, matches_var, corner_var):
    enable_bundle_adjustment, is_calibrate, show_matches, show_corners = get_ui_values(ba_var, data_var, matches_var, corner_var)
    
    if is_calibrate:
        K, image_list = calibrate_camera(show_corners)
    else:
        K, image_list = init()

    print(f"\nCamera Matrix (K):\n{K}")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # First camera pose: identity
    first_transform_mat = np.hstack((np.eye(3), np.zeros((3, 1))))
    first_pose = K @ first_transform_mat

    total_points = np.zeros((1, 3))
    total_colors = np.zeros((1, 3))
    #Adel
    #after Zaar code review
    _, second_image, first_feature, second_feature,E = select_image_pair(image_list, K, show_matches)


    print(f"\nFundamental Matrix with RANSAC:\n{E}")

    #Adel
    #Fixes after Zaar code review
    # Normalize matched points
    first_norm = cv2.undistortPoints(np.expand_dims(first_feature, axis=1), K, None)
    second_norm = cv2.undistortPoints(np.expand_dims(second_feature, axis=1), K, None)

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
    second_pose = K @ second_transform_mat

    # Triangulate 3D points
    first_feature, second_feature, points_3d = triangulation(first_pose, second_pose, first_feature, second_feature)

    # Reprojection error
    error, points_3d = reprojection_error(points_3d, second_feature, second_transform_mat, K, homogenity=1)
    print("First reprojection error:", error)

    # PnP for initial pose refinement
    _, _, second_feature, points_3d, _ = pnp_rasnac(
        points_3d,
        second_feature,
        K,
        np.zeros((5, 1), dtype=np.float32),
        first_feature,
        initial=1
    )

    # Bundle adjustment preparation
    total_images = len(image_list) - 2
    pose_array = np.hstack((K.ravel(), first_pose.ravel(), second_pose.ravel()))
    threshold = 0.5

    errors = []
    camera_poses = []

    # Loop through each subsequent image for incremental camera expansion
    for i in tqdm(range(total_images)):
        # Load and downscale current image for processing
        image_2 = downscale_image(cv2.imread(image_list[i + 2]))

        # Feature matching between previous and current image
        features_cur, features_2,_ = features_matching(second_image, image_2,show_matches,K)

        # If not the first iteration, perform triangulation for 3D points
        if i != 0:
            first_feature, second_feature, points_3d = triangulation(first_pose, second_pose, first_feature, second_feature)
            second_feature = second_feature.T
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
            points_3d = points_3d[:, 0, :]

        # Find common points between current and previous images
        cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = com_points(second_feature, features_cur, features_2,is_pre_calibrated=is_calibrate)
        
        cm_points_2 = features_2[cm_points_1]
        cm_points_cur = features_cur[cm_points_1]

        # Estimate the pose of the new camera using PnP
        rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = pnp_rasnac(points_3d[cm_points_0], cm_points_2, K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial=0)
        second_transform_mat = np.hstack((rot_matrix, tran_matrix))  # Combine rotation and translation into a single matrix
        pose_2 = np.matmul(K, second_transform_mat)  # Camera pose in world coordinates

        # Calculate reprojection error for the newly estimated 3D points
        error, points_3d = reprojection_error(points_3d, cm_points_2, second_transform_mat, K, homogenity=0)

        # Perform triangulation to estimate 3D points from the new camera pose
        cm_mask_0, cm_mask_1, points_3d = triangulation(second_pose, pose_2, cm_mask_0, cm_mask_1)
        error, points_3d = reprojection_error(points_3d, cm_mask_1, second_transform_mat, K, homogenity=1)
        print("rp error: ", error)

        # Add the new pose to the pose array
        pose_array = np.hstack((pose_array, pose_2.ravel()))
        # Store extrinsic pose as 4x4 for Open3D visualization
        Rt = np.hstack((rot_matrix, tran_matrix))  # 3x4
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :4] = Rt
        camera_poses.append(pose_4x4)  # Store for visualization

        # Perform bundle adjustment to refine 3D points and camera poses, if enabled
        if enable_bundle_adjustment:
            
            points_3d, cm_mask_1, second_transform_mat = bundle_adjustment(points_3d, cm_mask_1, second_transform_mat, K, threshold)
            pose_2 = np.matmul(K, second_transform_mat)  # Update the refined pose
            error, points_3d = reprojection_error(points_3d, cm_mask_1, second_transform_mat, K, homogenity=0)
            print("Bundle error: ", error)

            # Accumulate the refined 3D points and their colors
            total_points = np.vstack((total_points, points_3d))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([image_2[l[1], l[0]] for l in points_left])  # Store color data for points
            total_colors = np.vstack((total_colors, color_vector))
        else:
            # Without bundle adjustment, just accumulate the points
            total_points = np.vstack((total_points, points_3d[:, 0, :]))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])  # Store color data
            total_colors = np.vstack((total_colors, color_vector))

        # Prepare for the next iteration: update pose and feature points
        first_transform_mat = np.copy(second_transform_mat)
        first_pose = np.copy(second_pose)
        #to show latererrors
        errors.append(error)
        # Update images and feature sets for the next iteration
        first_image = np.copy(second_image)
        second_image = np.copy(image_2)
        first_feature = np.copy(features_cur)
        second_feature = np.copy(features_2)
        second_pose = np.copy(pose_2)

        # Display the current image and check for a break condition
        cv2.imshow(image_list[0].split('/')[-2], image_2)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # Close all OpenCV windows after processing is complete
    cv2.destroyAllWindows()
    
    show_results(errors,camera_poses,total_points,total_colors)
