import numpy as np
from skimage import measure,draw
from scipy.ndimage import map_coordinates
import math


def fspecial(func_name, sigma=1.,D=1):
    if func_name=='gaussian':
        if D ==2:
            m=n=max(1, np.fix(4*sigma))
            y,x=np.ogrid[-m:m+1,-n:n+1]
            h=np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        elif D==1:
            m = max(1, np.fix(4 * sigma))
            x = np.ogrid[-m:m + 1]
            h = 1/(np.sqrt(2*np.pi) * (sigma **3)) * (-x) * np.exp(-(x*x)/ (2*sigma*sigma))
        return h




def getMargin(img, mask, ScaleLength=6, pointN=15, stride=1):
    """
    :param img: 2D image, shape w x h
    :param mask: 2D mask, shape w x h
    :param ScaleLength: default is 6
    :param pointN: the width of margin
    :param stride: the number of points extracted from margin.
    :return:
    """

    img = np.pad(img, pad_width=((pointN, pointN), (pointN, pointN)), mode='constant')
    mask = np.pad(mask, pad_width=((pointN, pointN), (pointN, pointN)), mode='constant')
    contours = measure.find_contours(mask, 0.5)[0]  ## level is the smooth parameter, where near to 1, smooth less.

    g_filter = fspecial('gaussian',sigma=3/1.732)
    PtsBdary = contours.shape[0]
    x = contours[:, 0]
    y = contours[:, 1]
    xcords = np.concatenate([x[PtsBdary - ScaleLength : PtsBdary] ,x, x[0: ScaleLength]], axis=0)
    ycords = np.concatenate([y[PtsBdary - ScaleLength : PtsBdary] ,y, y[0: ScaleLength]], axis=0)
    if xcords.shape[0] <= g_filter.shape[0]:
        diffx = xcords
        diffy = ycords
    else:
        diffx = np.convolve(xcords, g_filter, mode='valid')
        diffy = np.convolve(ycords, g_filter, mode='valid')
    tanm = np.sqrt(diffx ** 2 + diffy ** 2)
    diffx = diffx / tanm
    diffy = diffy / tanm
    theta = np.arctan2(-diffx , diffy)

    L = np.ogrid[-pointN: pointN+1]
    p = np.zeros((math.ceil(PtsBdary / stride), 2 * pointN + 1))
    # start = time.time()
    k = 0
    for i in range(0, PtsBdary, stride):
        x1 = x[i] + L * np.cos(theta[i])
        y1 = y[i] + L * np.sin(theta[i])
        # Z = interp2d(img, xp=x1.tolist(), fp=y1.tolist())
        Z = map_coordinates(img, [x1, y1], order=1, mode='constant')
        InterZ = Z[0:pointN]
        OuterZ = Z[pointN + 1:2*pointN+1]
        if -np.pi < theta[i] < -np.pi / 2 or np.pi / 2 < theta[i] < np.pi:
            InterZ = Z[pointN + 1:2*pointN+1]
            InterZ = np.flip(InterZ)
            OuterZ = Z[0:pointN]
            OuterZ = np.flip(OuterZ)
        InterZ = InterZ.T
        OuterZ = OuterZ.T
        p[k, :] = np.concatenate([InterZ, Z[pointN:pointN + 1], OuterZ], axis=0)
        k += 1
    # print(time.time() - start)
    return p










