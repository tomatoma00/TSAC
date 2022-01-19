# this file is to test performance of the method
import cv2
import numpy as np



def basetest(loc1, loc2, H):
    loc1p = np.array([loc1[0], loc1[1], 1]).reshape(3, 1)
    newlocp = np.dot(H, loc1p)
    newlocq = newlocp.tolist()
    newloc = [newlocq[0][0], newlocq[1][0], newlocq[2][0]]
    if newloc[2] == 0:
        return 0
    newloc[0] = newloc[0] / newloc[2]
    newloc[1] = newloc[1] / newloc[2]
    newloc[2] = newloc[2] / newloc[2]
    error = (loc2[0] - newloc[0]) * (loc2[0] - newloc[0]) + (loc2[1] - newloc[1]) * (loc2[1] - newloc[1])
    # print(error)
    if error < 4:
        return 1
    else:
        return 0

def testscore(good, kp1, kp2, H):
    result = []
    for m in good:
        loc1 = [kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]]
        loc2 = [kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]
        result.append(basetest(loc1, loc2, H))
    # print(sum(result))
    return sum(result)


def testscoreimage(good, kp1, kp2, H, img1, img2):
    # visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    count = 0
    if H is None:
        print('not found')
        return 0
    for m in good:
        loc1 = [kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]]
        loc2 = [kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]
        result = basetest(loc1, loc2, H)
        if result == 1:
            color = (0, 240, 0)
            count = count + 1
        if result == 0:
            color = (0, 0, 240)
        # draw the keypoints
        cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
                 (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color, 2)
    view = cv2.resize(view, (int(0.5 * (w1 + w2)), int(0.5 * max(h1, h2))))
    print(count)
    cv2.imshow("view", view)
    cv2.waitKey()
