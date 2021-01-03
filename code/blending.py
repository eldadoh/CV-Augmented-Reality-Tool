
import numpy as np
import cv2
import scipy
from scipy.signal import convolve2d
import math

def split_rgb(image):
    (blue, green, red) = cv2.split(image)
    return red, green, blue

def generating_kernel(a):
    w_1d = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(w_1d, w_1d)

def ireduce(image):
    kernel = generating_kernel(0.5)
    outimage = scipy.signal.convolve2d(image, kernel, 'same')
    out = outimage[::2, ::2]
    return out

def iexpand(image):
    kernel = generating_kernel(0.5)
    outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = 4 * scipy.signal.convolve2d(outimage, kernel, 'same')
    return out

def gauss_pyramid(image, levels):
    output = []
    output.append(image)
    tmp = image
    for i in range(0, levels):
        tmp = ireduce(tmp)
        output.append(tmp)
    return output

def lapl_pyramid(gauss_pyr):
    output = []
    for i in range(0, len(gauss_pyr) - 1):
        gu = gauss_pyr[i]
        egu = iexpand(gauss_pyr[i + 1])
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(gauss_pyr.pop())
    return output

def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    blended_pyr = []
    for i in range(0,  len(gauss_pyr_mask)):
        p1 = gauss_pyr_mask[i] * lapl_pyr_white[i]
        p2 = (1 - gauss_pyr_mask[i]) * lapl_pyr_black[i]
        blended_pyr.append(p1 + p2)
    return blended_pyr

def collapse(lapl_pyr):
    output = np.zeros((lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr) - 1, 0, -1):
        lap = iexpand(lapl_pyr[i])
        lapb = lapl_pyr[i - 1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output


def blender(image1, image2, mask):


    (r1, g1, b1) = split_rgb(image1)
    (r2, g2, b2) = split_rgb(image2)
    (rm, gm, bm) = split_rgb(mask)

    rm = rm.astype(float) / 255
    gm = gm.astype(float) / 255
    bm = bm.astype(float) / 255

    # Automatically figure out the size
    min_size = min(r1.shape)
    depth = int(math.floor(math.log(min_size, 2))) - 4  # at least 16x16 at the highest level.

    gauss_pyr_maskr = gauss_pyramid(rm, depth)
    gauss_pyr_maskg = gauss_pyramid(gm, depth)
    gauss_pyr_maskb = gauss_pyramid(bm, depth)

    gauss_pyr_image1r = gauss_pyramid(r1, depth)
    gauss_pyr_image1g = gauss_pyramid(g1, depth)
    gauss_pyr_image1b = gauss_pyramid(b1, depth)

    gauss_pyr_image2r = gauss_pyramid(r2, depth)
    gauss_pyr_image2g = gauss_pyramid(g2, depth)
    gauss_pyr_image2b = gauss_pyramid(b2, depth)

    lapl_pyr_image1r = lapl_pyramid(gauss_pyr_image1r)
    lapl_pyr_image1g = lapl_pyramid(gauss_pyr_image1g)
    lapl_pyr_image1b = lapl_pyramid(gauss_pyr_image1b)

    lapl_pyr_image2r = lapl_pyramid(gauss_pyr_image2r)
    lapl_pyr_image2g = lapl_pyramid(gauss_pyr_image2g)
    lapl_pyr_image2b = lapl_pyramid(gauss_pyr_image2b)

    outimgr = collapse(blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr))
    outimgg = collapse(blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg))
    outimgb = collapse(blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb))

    result = np.zeros(image1.shape, dtype=image1.dtype)
    tmp = [outimgb, outimgg, outimgr]
    result = cv2.merge(tmp, result).astype(np.uint8)
    return result


if __name__ == '__main__':

    image1 = cv2.imread(r'C:\Users\hodp\Desktop\hw4_cv\hw4\code\test1.png')
    image2 = cv2.imread(r'C:\Users\hodp\Desktop\hw4_cv\hw4\code\test2.png')
    mask =np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    mask[0:image1.shape[0],image1.shape[1]//2:image1.shape[1]] = 255
    blender(image1, image2, mask)

#%% md

np.delete()