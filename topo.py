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
     edges = []
    edgenum = np.ones((len(points2)))
    prob = np.zeros((len(points2)))
    newprob = np.zeros((len(points2)))
    for s in sim:
        edge = [min(s[0],s[1]),max(s[0],s[1])]
        if not edge in edges:
            edges.append(edge)
            edgenum[s[0]] = edgenum[s[0]] + 1
            edgenum[s[1]] = edgenum[s[1]] + 1
        edge = [min(s[1],s[2]),max(s[1],s[2])]
        if not edge in edges:
            edges.append(edge)
            edgenum[s[1]] = edgenum[s[1]] + 1
            edgenum[s[2]] = edgenum[s[2]] + 1
        edge = [min(s[0],s[2]),max(s[0],s[2])]
        if not edge in edges:
            edges.append(edge)
            edgenum[s[0]] = edgenum[s[0]] + 1
            edgenum[s[2]] = edgenum[s[2]] + 1

    for i in range(0,len(edges)-1):
        A = points2[edges[i][0]]
        B = points2[edges[i][1]]
        for j in range(i+1,len(edges)):
            C = points2[edges[j][0]]
            D = points2[edges[j][1]]
            result = judgecross.judge(A, B, C, D)
            if result == 1:
                prob[edges[i][0]] = prob[edges[i][0]] + 1
                prob[edges[i][1]] = prob[edges[i][1]] + 1
                prob[edges[j][0]] = prob[edges[j][0]] + 1
                prob[edges[j][1]] = prob[edges[j][1]] + 1

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
