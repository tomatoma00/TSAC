# this file is to construct a topological network and reconnected network

import math
import cv2
import numpy as np
from scipy.spatial import Delaunay
import judgecross

def getdel(kp1, good):
    # get delaunary
    points = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points[i, 0] = kp1[m.queryIdx].pt[0]
        points[i, 1] = kp1[m.queryIdx].pt[1]
        i = i + 1
    tri = Delaunay(points)
    return [points, tri.simplices]

def changetype(kp2, good):
    points = np.zeros((len(good), 2))
    i = 0
    for m in good:
        points[i, 0] = kp2[m.trainIdx].pt[0]
        points[i, 1] = kp2[m.trainIdx].pt[1]
        i = i + 1
    return points

# draw the topology
def drawdel(points, sim, img1):
    trinum = np.shape(sim)[0]
    h1, w1 = img1.shape[:2]
    view = np.zeros((h1, w1, 3), np.uint8)
    view[:, :, 0] = img1
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    for i in range(0, trinum):
        cv2.line(view, (int(points[sim[i][0]][0]), int(points[sim[i][0]][1])),
                 (int(points[sim[i][1]][0]), int(points[sim[i][1]][1])), (0, 180, 180), 2)
        cv2.line(view, (int(points[sim[i][0]][0]), int(points[sim[i][0]][1])),
                 (int(points[sim[i][2]][0]), int(points[sim[i][2]][1])), (0, 180, 180), 2)
        cv2.line(view, (int(points[sim[i][1]][0]), int(points[sim[i][1]][1])),
                 (int(points[sim[i][2]][0]), int(points[sim[i][2]][1])), (0, 180, 180), 2)
    cv2.imshow('tri', view)
    cv2.waitKey(0)

# get average cross time
def gettimes2(sim,points2):
    trinum = np.shape(sim)[0]
    edgenum = np.ones((len(points2)))
    prob = np.zeros((len(points2)))
    newprob = np.zeros((len(points2)))
    index = [[0,1],[1,2],[0,2]]
    for i in range(0,trinum):
        # 第一层循环
        for j in index:
            A = points2[sim[i][j[0]]]
            B = points2[sim[i][j[1]]]
            for k in range(i+1,trinum):
                for l in index:
                    C = points2[sim[k][l[0]]]
                    D = points2[sim[k][l[1]]]
                    result = judgecross.judge(A,B,C,D)
                    if result == 1:
                        prob[sim[i][j[0]]] = prob[sim[i][j[0]]] + 1
                        prob[sim[i][j[1]]] = prob[sim[i][j[1]]] + 1
                        prob[sim[k][l[0]]] = prob[sim[k][l[0]]] + 1
                        prob[sim[k][l[1]]] = prob[sim[k][l[1]]] + 1
    for s in sim:
        edgenum[s[0]] = edgenum[s[0]] + 1
        edgenum[s[1]] = edgenum[s[1]] + 1
        edgenum[s[2]] = edgenum[s[2]] + 1
    secondorder=0
    for i in range(0,len(points2)):
        if edgenum[i]==0:
            prob[i] = 99
        else:
            prob[i]=prob[i]/edgenum[i]
        secondorder = secondorder + prob[i] * prob[i]
    # print(prob)
    secondorder = math.sqrt(secondorder/len(points2))
    for i in range(0,len(points2)):
        # newprob[i]=1-(1/(math.sqrt(2*math.pi)*secondorder))*math.exp(-prob[i]*prob[i]/(2*secondorder*secondorder))
        newprob[i]=1-math.exp(-prob[i]*prob[i]/(2*secondorder*secondorder))
    return newprob


