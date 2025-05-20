#mariam
import numpy as np


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
