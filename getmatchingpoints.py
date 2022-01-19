# this file is to get keypoints and match them

import cv2
import scipy as sp
import numpy as np


def siftandgetlocation(img1,img2,goodflag=1):
    sift = cv2.xfeatures2d.SIFT_create(600)

    # find the keypoints and descriptors with SIFT
    loc1, des1 = sift.detectAndCompute(img1, None)
    loc2, des2 = sift.detectAndCompute(img2, None)
    # abb = cv2.drawKeypoints(img2, loc2, img2, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  #
    # cv2.imshow('abb', abb)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print('num of matching points', len(good))

    return [good,loc1,loc2]


# show the image matches
def showtheimage(good,kp1,kp2,img1,img2,ratio = 0.5):
    # visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    if good is None:
        view = cv2.resize(view, (int(ratio * (w1 + w2)), int(ratio * max(h1, h2))))
        cv2.imshow("view", view)

        cv2.waitKey()
        return

    for m in good:
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple([sp.random.randint(0, 255) for _ in range(3)])
        # print 'kp1,kp2',kp1,kp2
        cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
                 (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color,2)
    view = cv2.resize(view,(int(ratio*(w1 + w2)),int(ratio*max(h1, h2))))
    cv2.imshow("view", view)

    cv2.waitKey()

# get keypoints which is included in list 'good'
def getnewkp(good,kp1,kp2):
    newkp1 =[]
    newkp2 =[]
    for m in good:
        newkp1.append(kp1[m.queryIdx])
        newkp2.append(kp2[m.trainIdx])
    return newkp1,newkp2

if __name__ =='__main__':
    img1 = cv2.imread('01.png', 0)
    img2 = cv2.imread('02.png', 0)
    result = siftandgetlocation(img1, img2, 1)
    good = result[0]
    kp1 = result[1]
    kp2 = result[2]
    showtheimage(good,kp1,kp2,img1,img2)
