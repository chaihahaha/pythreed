import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
from PIL import Image

def matching_points(img_name1, img_name2):
    img1 = cv.imread(img_name1,cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(img_name2,cv.IMREAD_GRAYSCALE) # trainImage
    
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=10000)#, nlevels=10, patchSize=20)
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
     
    # Match descriptors.
    matches = bf.match(des1,des2)
     
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
     
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = Image.fromarray(img3)
    img3.save('match10.png')

    points1 = []
    points2 = []
    for m in matches:
        points1.append(list(kp1[m.queryIdx].pt))
        points2.append(list(kp2[m.trainIdx].pt))
    points1 = np.array(points1)
    points2 = np.array(points2)
     
    return points1, points2


def matching_points_LoFTR(img_name1, img_name2):
    return points1, points2
