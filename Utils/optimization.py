import cv2
import numpy as np
from scipy.optimize import least_squares

def opt_reprojection_error(obj_points, dist_coeff, fix_K, fixed_K=None) -> np.array:
    transf_mat = obj_points[0:12].reshape((3, 4))
    if fix_K:
        # Use fixed K passed externally
        K = fixed_K
        rest = int(len(obj_points[12:]) * 0.4)
        p = obj_points[12:12 + rest].reshape((2, int(rest / 2))).T
        obj_points_3d = obj_points[12 + rest:].reshape((-1, 3))
    else:
        K = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
        obj_points_3d = obj_points[21 + rest:].reshape((-1, 3))

    rot_m = transf_mat[:, :3]
    tran_vector = transf_mat[:, 3]
    rot_vector, _ = cv2.Rodrigues(rot_m)

    image_points, _ = cv2.projectPoints(obj_points_3d, rot_vector, tran_vector, K, dist_coeff)
    image_points = image_points[:, 0, :]
    error = [(p[i] - image_points[i])**2 for i in range(len(p))]
    return np.array(error).ravel() / len(p)

def bundle_adjustment(_3d_point, opt, transform_matrix_new, K, r_error, dist_coeff, is_calibrate) -> tuple:
    fix_K = not is_calibrate  # Fix intrinsic matrix K for pre-calibrated datasets

    variables = [transform_matrix_new.ravel()]
    if not fix_K:
        variables.append(K.ravel())
    variables.append(opt.ravel())
    variables.append(_3d_point.ravel())
    opt_variables = np.hstack(variables)

    if fix_K:
        # Pass fixed K to the error function via a lambda closure
        result = least_squares(
            lambda op: opt_reprojection_error(op, dist_coeff, fix_K=True, fixed_K=K),
            opt_variables,
            gtol=r_error,
            loss='huber',
            method='trf'
        ).x
        offset = 12
    else:
        result = least_squares(
            lambda op: opt_reprojection_error(op, dist_coeff, fix_K=False),
            opt_variables,
            gtol=r_error,
            loss='huber',
            method='trf'
        ).x
        offset = 21
        K = result[12:21].reshape((3, 3))

    rest = int(len(result[offset:]) * 0.4)
    points_2d = result[offset:offset + rest].reshape((2, int(rest / 2))).T
    points_3d = result[offset + rest:].reshape((-1, 3))
    transform = result[0:12].reshape((3, 4))

    return points_3d, points_2d, transform, K
