import os

import cv2
from config import (
    reduce_res_fac,
    is_down_scale_images
)


def get_images(path, extension):
    image_list = []
    for image_name in sorted(os.listdir(path)):
        if image_name[-4:].lower() == extension:
            full_path = os.path.join(path, image_name)
            img = cv2.imread(full_path)
            if img is not None:
                image_list.append(img)
    return image_list

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