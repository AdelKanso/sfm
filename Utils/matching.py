#mariam
import cv2
import numpy as np

from Plot.plot import show_image


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
