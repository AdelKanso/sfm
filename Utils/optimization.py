import cv2
import numpy as np
from scipy.optimize import least_squares

def opt_reprojection_error(obj_points, dist_coeff, fix_K, fixed_K=None) -> np.array:
    # --- Parse inputs ---
    # Extract transformation matrix (rotation + translation)
    transf_mat = obj_points[0:12].reshape((3, 4))
    
    # Depending on whether intrinsic matrix K is fixed or optimized,
    # extract K, 2D projected points, and 3D object points accordingly
    if fix_K:
        K = fixed_K
        rest = int(len(obj_points[12:]) * 0.4)
        p = obj_points[12:12 + rest].reshape((2, int(rest / 2))).T
        obj_points_3d = obj_points[12 + rest:].reshape((-1, 3))
    else:
        K = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
        obj_points_3d = obj_points[21 + rest:].reshape((-1, 3))

    # --- Convert rotation matrix to vector form ---
    rot_m = transf_mat[:, :3]
    tran_vector = transf_mat[:, 3]
    rot_vector, _ = cv2.Rodrigues(rot_m)

    # --- Project 3D points into image plane ---
    image_points, _ = cv2.projectPoints(obj_points_3d, rot_vector, tran_vector, K, dist_coeff)
    image_points = image_points[:, 0, :]

    # --- Compute reprojection error ---
    error = [(p[i] - image_points[i])**2 for i in range(len(p))]

    # Return normalized reprojection error vector for optimization
    return np.array(error).ravel() / len(p)


def bundle_adjustment(_3d_point, opt, transform_matrix_new, K, r_error, dist_coeff, is_calibrate) -> tuple:
    # --- Setup optimization variables ---
    # Decide whether intrinsic matrix K is fixed or should be optimized
    fix_K = not is_calibrate

    # Collect variables to optimize: transform, K (optional), distortion params, 3D points
    variables = [transform_matrix_new.ravel()]
    if not fix_K:
        variables.append(K.ravel())
    variables.append(opt.ravel())
    variables.append(_3d_point.ravel())
    opt_variables = np.hstack(variables)

    # --- Run least squares optimization ---
    # Use a robust loss (Huber) to reduce effect of outliers
    if fix_K:
        # Optimize with fixed K, pass K explicitly
        result = least_squares(
            lambda op: opt_reprojection_error(op, dist_coeff, fix_K=True, fixed_K=K),
            opt_variables,
            gtol=r_error,
            loss='huber',
            method='trf'
        ).x
        offset = 12  # index offset for parsing results
    else:
        # Optimize including K
        result = least_squares(
            lambda op: opt_reprojection_error(op, dist_coeff, fix_K=False),
            opt_variables,
            gtol=r_error,
            loss='huber',
            method='trf'
        ).x
        offset = 21
        K = result[12:21].reshape((3, 3))  # update optimized K

    # --- Extract optimized results ---
    rest = int(len(result[offset:]) * 0.4)
    points_2d = result[offset:offset + rest].reshape((2, int(rest / 2))).T
    points_3d = result[offset + rest:].reshape((-1, 3))
    transform = result[0:12].reshape((3, 4))

    # Return updated 3D points, 2D points, transformation matrix, and intrinsics
    return points_3d, points_2d, transform, K
