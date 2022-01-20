# this file is to judge whether two edges cross
import numpy as np

ab = np.array([0, 0])
ac = np.array([0, 0])
ad = np.array([0, 0])
cd = np.array([0, 0])
ca = np.array([0, 0])
cb = np.array([0, 0])


def judge(line1a, line1b, line2c, line2d):
    if max(line1a[0], line1b[0]) <= min(line2c[0], line2d[0]):
        return 0
    elif max(line2c[0], line2d[0]) <= min(line1a[0], line1b[0]):
        return 0
    elif max(line1a[1], line1b[1]) <= min(line2c[1], line2d[1]):
        return 0
    elif max(line2c[1], line2d[1]) <= min(line1a[1], line1b[1]):
        return 0
    else:
        ab[0], ab[1] = line1b[0] - line1a[0], line1b[1] - line1a[1]
        ac[0], ac[1] = line2c[0] - line1a[0], line2c[1] - line1a[1]
        ad[0], ad[1] = line2d[0] - line1a[0], line2d[1] - line1a[1]
        cd[0], cd[1] = line2d[0] - line2c[0], line2d[1] - line2c[1]
        ca[0], ca[1] = line1a[0] - line2c[0], line1a[1] - line2c[1]
        cb[0], cb[1] = line1b[0] - line2c[0], line1b[1] - line2c[1]
        result1 = np.dot(np.cross(ab, ac), np.cross(ab, ad))
        result2 = np.dot(np.cross(cd, ca), np.cross(cd, cb))
        if result2 < 0 and result1 < 0:
            return 1
        else:
            return 0
