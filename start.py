import cv2
import numpy as np
import getmatchingpoints
import testmodel
import topo
import TSAC

img1 = cv2.imread('01.png', 0)
img2 = cv2.imread('02.png', 0)
Hlist = [[1 ,0, -20],[0, 1, -569],[0 ,0 ,1]]
H = np.array(Hlist)
result = getmatchingpoints.siftandgetlocation(img1, img2, 1)

good = result[0]
kp1 = result[1]
kp2 = result[2]
print('the correct matches before removal:')
testmodel.testscoreimage(good, kp1, kp2, H, img1, img2)
# main.showtheimage(good,kp1,kp2,img1,img2)

result1 = topo.getdel(kp1, good)
points1 = result1[0]
points2 = topo.changetype(kp2, good)
sim = result1[1]
topo.drawdel(points1, sim, img1)
topo.drawdel(points2, sim, img2)

p = topo.gettimes2(sim, points2)
newkp1, newkp2 = getmatchingpoints.getnewkp(good, kp1, kp2)

afterm = TSAC.ransacp(good, newkp1, newkp2, p, kp1, kp2)
print('the number of matches after removal:', len(afterm))
summ = testmodel.testscore(afterm, kp1, kp2, H)
print('the number of correct matches after removal:', summ)
# testratio.testscoreimage(afterm, kp1, kp2, H,img1,img2)
