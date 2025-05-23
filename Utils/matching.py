#mariam
import cv2
import numpy as np

from Plot.plot import show_image

from Utils.io_utils import downscale_image

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
    print(f"good matches {len(feature)}")
    pts0=np.float32([first_key_points[m.queryIdx].pt for m in feature])
    pts1=np.float32([second_key_points[m.trainIdx].pt for m in feature])
    # Find Fundamental Matrix with RANSAC
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)

    #calculate essential matrix
    E = K.T @ F @ K
    # Return coordinates of matched keypoints
    # Select only inlier founded_matches
    return pts0[mask.ravel() == 1],pts1[mask.ravel() == 1],E


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