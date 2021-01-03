from __future__ import print_function
from PIL import Image
import numpy as np
import cv2.cv2
from matplotlib import pyplot as plt
import os
import random
from scipy import interpolate
from blending import blender

def plot_images(images, title):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
    fig.suptitle(title)
    for i in range(len(images)):
        # axes[idx].title.set_text()
        axes[i].imshow(images[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()


def convert_PIL_to_cv2(PILimage):
    pil_image = PILimage.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def read_images(path):
    """
    :param path: path of folder
    :return: list of images
    """
    files = os.listdir(path)
    pics = []
    for i in range(len(files)):
        pics.append(Image.open(path + "/" + str(i) + files[i][-4:]))

    return pics


# Extra functions end

# HW functions:
def getPoints(im1, im2, N):
    plt.figure()
    plt.imshow(im1)
    p1 = plt.ginput(N, timeout=150)
    plt.close()
    print(p1)
    plt.figure()
    plt.imshow(im2)
    p2 = plt.ginput(N, timeout=150)
    plt.close()
    return np.asarray(p1).T, np.asarray(p2).T


def computeH(list1, list2):
    list1 = list1.T
    list2 = list2.T
    A = np.zeros([len(list1) * 2, 9])
    x = 0
    # compute matrix A
    for i in range(len(list1)):
        A[x, 0] = list1[i][0]
        A[x, 1] = list1[i][1]
        A[x, 2] = 1
        A[x, 6] = -list1[i][0] * list2[i][0]
        A[x, 7] = -list1[i][1] * list2[i][0]
        A[x, 8] = -list2[i][0]
        A[x + 1][3] = list1[i][0]
        A[x + 1][4] = list1[i][1]
        A[x + 1][5] = 1
        A[x + 1][6] = -list1[i][0] * list2[i][1]
        A[x + 1][7] = -list1[i][1] * list2[i][1]
        A[x + 1][8] = -list2[i][1]
        x = x + 2
    [U, S, V] = np.linalg.svd(A, False)
    m = V.T[:, -1]
    H2to1 = np.reshape(m, (3, 3))
    H2to1 = H2to1
    return H2to1


def check_homography(points1, points2, im1, im2, title):
    for p1, p2 in zip(points1.T, points2.T):
        # color  = np.random.choice(range(255),size=3)
        color = np.random.randint(0, 255, (1, 3))
        color = (list(np.random.choice(range(256), size=3)))
        color = [int(color[0]), int(color[1]), int(color[2])]
        cv2.circle(im1, tuple((int(p1[1]), int(p1[0]))), 10, color, -1)
        cv2.circle(im2, tuple((int(p2[0]), int(p2[1]))), 10, color, -1)
    images = [im1, im2]
    plot_images(images, title)
    return im1, im2

def check_homography_calculate(points1, points2, im1, im2, title):
    for p1, p2 in zip(points1.T, points2.T):
        # color  = np.random.choice(range(255),size=3)
        color = np.random.randint(0, 255, (1, 3))
        color = (list(np.random.choice(range(256), size=3)))
        color = [int(color[0]), int(color[1]), int(color[2])]
        cv2.circle(im1, tuple((int(p1[0]), int(p1[1]))), 5, color, -1)
        cv2.circle(im2, tuple((int(p2[0]), int(p2[1]))), 5, color, -1)
    images = [im1, im2]
    plot_images(images, title)
    return im1, im2


def calculate_size(size_image1, size_image2, H):
    corners = np.array([[0, 0, 1],
                        [0, size_image1[0], 1],
                        [size_image1[1], size_image1[0], 1],
                        [size_image1[1], 0, 1]])
    points = []
    points = np.zeros((8, 2))
    for i in range(len(corners)):
        x, y, z = np.dot(((H)), np.array([[corners[i][0]], [corners[i][1]], [1]]))
        pixel_x, pixel_y = [int(x / z), int(y / z)]
        points[i][0] = pixel_x
        points[i][1] = pixel_y
    points[4] = [0, 0]
    points[5] = [0, size_image2[0]]
    points[6] = [size_image2[1], size_image2[0]]
    points[7] = [size_image2[1], 0]
    min_axisX = min(points[:, 0])
    max_axisX = max(points[:, 0])
    min_axisY = min(points[:, 1])
    max_axisY = max(points[:, 1])
    offset_x = 0
    offset_y = 0
    if min_axisX < 0:
        offset_x = abs(min_axisX)
    if min_axisY < 0:
        offset_y = abs(min_axisY)
    width_X = abs(min_axisX) + max_axisX
    length_Y = abs(min_axisY) + max_axisY
    offset = int(offset_x), int(offset_y)
    panorma_size = int(width_X), int(length_Y)
    T = np.identity(3)
    T[0][2] = offset[0]
    T[1][2] = offset[1]
    # calculate the new H matrixs
    H = np.dot(T, H)
    return offset, panorma_size, H


def cell_neighbors(i, j, img):
    x = np.array((i - 1, i - 1, i - 1, i, i + 1, i + 1, i + 1, i))
    y = np.array((j - 1, j, j + 1, j + 1, j + 1, j, j - 1, j - 1))
    z = []
    for i in range(len(x)):
        try:
            z.append(img[x[i],y[i]])
        except:
            z.append(img[i,j])
    return x,y,z
# def get_colors_for_interpolate(xx,yy):
#     colors = np.zeros((len(xx),len(yy)))
#     for i in range(len(xx)):
#         for j in range()
def warpH(im1, H, out_size):
    print("warpH")
    xx, yy = np.meshgrid(np.arange(0,im1.shape[1]), np.arange(0,im1.shape[0]))
    interpolate_r = interpolate.interp2d(np.arange(0, im1.shape[1]), np.arange(0, im1.shape[0]), im1[yy, xx][:, :, 0], kind='linear')
    interpolate_g = interpolate.interp2d(np.arange(0, im1.shape[1]), np.arange(0, im1.shape[0]), im1[yy, xx][:, :, 1], kind='linear')
    interpolate_b = interpolate.interp2d(np.arange(0, im1.shape[1]), np.arange(0, im1.shape[0]), im1[yy, xx][:, :, 2], kind='linear')
    # warp_im1 = cv2.warpPerspective(im1, H, out_size)
    warp_im1 = np.zeros([out_size[1], out_size[0], 3], dtype=np.uint8)
    for i in range(1, out_size[0]):
        for j in range(1, out_size[1]):
            x, y, z = np.dot(np.linalg.inv(H), np.array([[i], [j], [1]]))
            if x/z >= 0 and x/z < (len(im1[0])) and y/z >= 0 and y/z < (len(im1)):
                warp_im1[j, i] = [int(interpolate_r(x/z, y/z)[0]),int(interpolate_g(x/z, y/z)[0]),int(interpolate_b(x/z, y/z)[0])]
    return warp_im1


def imageStitching(img1, wrap_img2, offset):
    print("imageStitching")
    panoImg = wrap_img2
    (h1, w1) = img1.shape[:2]
    for h in range(h1):
        for w in range(w1):
            if img1[h][w][0] != 0 or img1[h][w][1] != 0 or img1[h][w][2] != 0:
                try:
                    panoImg[h + int(offset[1])][w + int(offset[0])] = img1[h][w]
                except:
                    pass
    return panoImg

def imageStitching_for_blender(img1, wrap_img2, offset):
    print("imageStitching")
    panoImg = np.zeros((wrap_img2.shape[0],wrap_img2.shape[1],3),dtype=np.uint8)
    (h1, w1) = img1.shape[:2]
    for h in range(h1):
        for w in range(w1):
            if img1[h][w][0] != 0 or img1[h][w][1] != 0 or img1[h][w][2] != 0:
                try:
                    panoImg[h + int(offset[1])][w + int(offset[0])] = img1[h][w]
                except:
                    pass
    return panoImg

def ransacH(p1, p2, nIter, tol):
    maxInliers = []
    bestH = None
    best_inliers_p1 = None
    best_inliers_p2 = None
    for i in range(nIter):
        # find 4 random points to calculate a homography
        rand_number = random.sample(range(0, len(p1[0])), 4)
        list1 = np.asarray(
            [p1[:, rand_number[0]], p1[:, rand_number[1]], p1[:, rand_number[2]], p1[:, rand_number[3]]]).T
        list2 = np.asarray(
            [p2[:, rand_number[0]], p2[:, rand_number[1]], p2[:, rand_number[2]], p2[:, rand_number[3]]]).T
        # call the homography function on those points
        h = computeH(list1, list2)
        inliers = []
        inliers_p1 = []
        inliers_p2 = []
        for ind in range(len(p1[0])):
            d = geometricDistance(p1.T[ind], p2.T[ind], h)
            if d < 5:
                inliers.append(1)
                inliers_p1.append(p1.T[ind])
                inliers_p2.append(p2.T[ind])
        # print(len(inliers))
        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            best_inliers_p1 = np.asarray(inliers_p1).T
            best_inliers_p2 = np.asarray(inliers_p2).T
            bestH = computeH(np.asarray(inliers_p1).T, np.asarray(inliers_p2).T)
            # bestH = h

        # if len(maxInliers) > (len(p1[0]) * tol):
        #     break
    check_homography_calculate(best_inliers_p1, best_inliers_p2, beach5, beach4, "points after ransac")
    return bestH


def geometricDistance(p1, p2, h):
    p1 = np.array((p1[0], p1[1], 1)).reshape((3, 1))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.array((p2[0], p2[1], 1)).reshape((3, 1))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def getPoints_SIFT(im1, im2):
    p1 = []
    p2 = []
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            p1.append((x1, y1))
            p2.append((x2, y2))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img_match = np.zeros((im1.shape[0] + im2.shape[0], im1.shape[1] + im2.shape[1]))
    img_match = cv2.drawMatchesKnn(im1, kp1, im2, kp2, good, img_match, flags=2)
    # print(len(p1))
    plt.imshow(img_match), plt.show()
    p1 = np.asarray(p1).T
    p2 = np.asarray(p2).T
    # p1, p2 = get_coordinate_from_sift(matches, kp1, kp2)
    return p1, p2


def build_sintra_panorama_sift(sintra1, sintra2, sintra3, sintra4, sintra5, ransac=False):
    # sintra 1_2
    p1, p2 = getPoints_SIFT(sintra1, sintra2)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra1.shape, sintra2.shape, H)
    warp_sintra1 = warpH(sintra1, H, panorma_size)
    sintra_1_2 = imageStitching(sintra2, warp_sintra1, offset)

    # #sintra 4_5
    p1, p2 = getPoints_SIFT(sintra4, sintra5)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra4.shape, sintra5.shape, H)
    warp_sintra4 = warpH(sintra4, H, panorma_size)
    sintra_4_5 = imageStitching(sintra5, warp_sintra4, offset)

    # sintra 3_4_5
    p1, p2 = getPoints_SIFT(sintra_4_5, sintra3)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra_4_5.shape, sintra3.shape, H)
    warp_sintra4_5 = warpH(sintra_4_5, H, panorma_size)
    sintra_3_4_5 = imageStitching(sintra3, warp_sintra4_5, offset)

    # sintra 1_2_3_4_5
    p1, p2 = getPoints_SIFT(sintra_1_2, sintra_3_4_5)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra_1_2.shape, sintra_3_4_5.shape, H)
    warp_sintra1_2 = warpH(sintra_1_2, H, panorma_size)
    sintra_1_2_3_4_5 = imageStitching(sintra_3_4_5, warp_sintra1_2, offset)

    return sintra_1_2_3_4_5


def build_beach_panorama_sift(beach1, beach2, beach3, beach4, beach5, ransac=False):
    # # beach 4_5
    # p1, p2 = getPoints_SIFT(beach1, beach2)
    # H = computeH(p1, p2)
    # if ransac:
    #     H = ransacH(p1, p2, 1000, 0.9)
    # offset, panorma_size, H = calculate_size(beach1.shape, beach2.shape, H)
    # warp_beach1 = warpH(beach1, H, panorma_size)
    # beach_1_2 = imageStitching(beach2, warp_beach1, offset)

    # beach 4_5
    p1, p2 = getPoints_SIFT(beach5, beach4)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.9)
    offset, panorma_size, H = calculate_size(beach5.shape, beach4.shape, H)
    warp_beach5 = warpH(beach5, H, panorma_size)
    beach_4_5 = imageStitching(beach4, warp_beach5, offset)

    # #beach 2_3
    p1, p2 = getPoints_SIFT(beach3, beach2)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.9)
    offset, panorma_size, H = calculate_size(beach3.shape, beach2.shape, H)
    warp_beach3 = warpH(beach3, H, panorma_size)
    beach_2_3 = imageStitching(beach2, warp_beach3, offset)

    # sintra 2_3_4_5
    p1, p2 = getPoints_SIFT(beach_2_3, beach_4_5)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.9)
    offset, panorma_size, H = calculate_size(beach_2_3.shape, beach_4_5.shape, H)
    warp_beach2_3 = warpH(beach_2_3, H, panorma_size)
    beach_2_3_4_5 = imageStitching(beach_4_5, warp_beach2_3, offset)

    return beach_2_3_4_5


def build_our_panorama (left,middle,right,ransac = False ):
    p1, p2 = getPoints_SIFT( left,middle )
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 10000, 0.9)
    offset, panorma_size, H = calculate_size(left.shape, middle.shape, H)
    warp_left = warpH(left, H, panorma_size)
    left_middle = imageStitching(middle, warp_left, offset)


    p1, p2 = getPoints_SIFT(right, left_middle)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 10000, 0.99)
    offset, panorma_size, H = calculate_size(right.shape, left_middle.shape, H)
    warp_right = warpH(right, H, panorma_size)
    panorama  = imageStitching(left_middle, warp_right, offset)
    return panorama

def build_our_panorama_blender(left,middle,right,ransac = False ):
    p1, p2 = getPoints_SIFT( left,middle )
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 10000, 0.9)
    offset, panorma_size, H = calculate_size(left.shape, middle.shape, H)
    warp_left = warpH(left, H, panorma_size)
    left_middle = imageStitching_for_blender(middle, warp_left, offset)
    mask =np.zeros((left_middle.shape[0], left_middle.shape[1], 3), dtype=np.uint8)
    mask[0:left_middle.shape[0],left_middle.shape[1]//2:left_middle.shape[1]] = 255
    left_middle = blender(warp_left, left_middle, mask)

    p1, p2 = getPoints_SIFT(right, left_middle)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 10000, 0.99)
    offset, panorma_size, H = calculate_size(right.shape, left_middle.shape, H)
    warp_right = warpH(right, H, panorma_size)
    panorama = imageStitching_for_blender(left_middle, warp_right, offset)
    mask =np.zeros((panorama.shape[0], panorama.shape[1], 3), dtype=np.uint8)
    mask[0:panorama.shape[0],panorama.shape[1]//2:panorama.shape[1]] = 255
    panorama = blender(panorama, warp_right, mask)
    return panorama
def build_sintra_panorama_manual(sintra1, sintra2, sintra3, sintra4, sintra5, ransac=False):
    # sintra 1_2
    p1, p2 = getPoints(sintra1, sintra2, 6)
    H = computeH(p1, p2)
    print (H)
    print("h1to2")

    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra1.shape, sintra2.shape, H)
    warp_sintra1 = warpH(sintra1, H, panorma_size)
    sintra_1_2 = imageStitching(sintra2, warp_sintra1, offset)

    # #sintra 4_5
    p1, p2 = getPoints(sintra4, sintra5, 6)
    H = computeH(p1, p2)

    print("h4to5")
    print(H)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra4.shape, sintra5.shape, H)
    warp_sintra4 = warpH(sintra4, H, panorma_size)
    sintra_4_5 = imageStitching(sintra5, warp_sintra4, offset)

    # sintra 3_4_5
    p1, p2 = getPoints(sintra_4_5, sintra3, 6)
    H = computeH(p1, p2)
    print("h45to3")
    print(H)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra_4_5.shape, sintra3.shape, H)
    warp_sintra4_5 = warpH(sintra_4_5, H, panorma_size)
    sintra_3_4_5 = imageStitching(sintra3, warp_sintra4_5, offset)

    # sintra 1_2_3_4_5
    p1, p2 = getPoints(sintra_1_2, sintra_3_4_5, 6)
    H = computeH(p1, p2)
    print("h12to345")
    print(H)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.99)
    offset, panorma_size, H = calculate_size(sintra_1_2.shape, sintra_3_4_5.shape, H)
    warp_sintra1_2 = warpH(sintra_1_2, H, panorma_size)
    sintra_1_2_3_4_5 = imageStitching(sintra_3_4_5, warp_sintra1_2, offset)


def build_beach_panorama_manual(beach1, beach2, beach3, beach4, beach5, ransac=False):
    # # beach 4_5
    # p1, p2 = getPoints_SIFT(beach1, beach2)
    # H = computeH(p1, p2)
    # if ransac:
    #     H = ransacH(p1, p2, 1000, 0.9)
    # offset, panorma_size, H = calculate_size(beach1.shape, beach2.shape, H)
    # warp_beach1 = warpH(beach1, H, panorma_size)
    # beach_1_2 = imageStitching(beach2, warp_beach1, offset)
    np.zeros_like()
    # beach 4_5
    p1, p2 = getPoints(beach5, beach4, 6)
    H = computeH(p1, p2)
    print(H)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.9)
    offset, panorma_size, H = calculate_size(beach5.shape, beach4.shape, H)
    warp_beach5 = warpH(beach5, H, panorma_size)
    beach_4_5 = imageStitching(beach4, warp_beach5, offset)

    # #beach 2_3
    p1, p2 = getPoints(beach3, beach2, 6)
    print(H)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.9)
    offset, panorma_size, H = calculate_size(beach3.shape, beach2.shape, H)
    warp_beach3 = warpH(beach3, H, panorma_size)
    beach_2_3 = imageStitching(beach2, warp_beach3, offset)

    # sintra 2_3_4_5
    p1, p2 = getPoints(beach_2_3, beach_4_5, 6)
    print(H)
    H = computeH(p1, p2)
    if ransac:
        H = ransacH(p1, p2, 5000, 0.9)
    offset, panorma_size, H = calculate_size(beach_2_3.shape, beach_4_5.shape, H)
    warp_beach2_3 = warpH(beach_2_3, H, panorma_size)
    beach_2_3_4_5 = imageStitching(beach_4_5, warp_beach2_3, offset)

    return beach_2_3_4_5

if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    im1 = cv2.resize(im1, (int(im1.shape[1] / 2), int(im1.shape[0] / 2)))
    im2 = cv2.resize(im2, (int(im2.shape[1] / 2), int(im2.shape[0] / 2)))

    ##Q 2.1
    p1, p2 = getPoints(im1, im2, 6)
    # check_homography(p1, p2, im1, im2, "points for homography")
    # Q 2.2
    H = computeH(p1, p2)

    #check for calculating homography
    p1 = np.asarray([[398, 167], [255,56],[203,92], [218,160],[357,147],[244,30]]).T
    p2 = np.zeros((2,p1.shape[1]))
    index = 0
    for p in p1.T:
        x, y, z = np.dot(H, np.array([[p[0]], [p[1]], [1]]))
        p2[:,index] = np.asarray([[int(x / z), int(y / z)]])
        index+=1
    check_homography_calculate(p1, p2, im1, im2, "points for homography")

    # 2.3+2.4
    offset, panorma_size, H = calculate_size(im1.shape, im2.shape, H)
    warp_im1 = warpH(im1, H, panorma_size)
    panorama = imageStitching(im2, warp_im1, offset)
    cv2.imwrite('panorama.jpg', panorama)
    cv2.imshow('img', panorama)

    #2.5
    p1, p2 = getPoints_SIFT(im1,im2)
    H = computeH(p1, p2)
    offset, panorma_size, H = calculate_size(im1.shape, im2.shape, H)
    warp_im1 = warpH(im1, H, panorma_size)

    panorama = imageStitching(im2, warp_im1, offset)
    cv2.imwrite('panorama.jpg', panorama)

    # #2.7 + 2.8
    # #sintra panorama
    sintra1 = cv2.imread('data/sintra1.jpg')
    sintra2 = cv2.imread('data/sintra2.jpg')
    sintra3 = cv2.imread('data/sintra3.jpg')
    sintra4 = cv2.imread('data/sintra4.jpg')
    sintra5 = cv2.imread('data/sintra5.jpg')
    sintra1 = cv2.resize(sintra1, (int(sintra1.shape[1] / 7), int(sintra1.shape[0] / 7)))
    sintra2 = cv2.resize(sintra2, (int(sintra2.shape[1] / 7), int(sintra2.shape[0] / 7)))
    sintra3 = cv2.resize(sintra3, (int(sintra3.shape[1] / 7), int(sintra3.shape[0] / 7)))
    sintra4 = cv2.resize(sintra4, (int(sintra4.shape[1] / 7), int(sintra4.shape[0] / 7)))
    sintra5 = cv2.resize(sintra5, (int(sintra5.shape[1] / 7), int(sintra5.shape[0] / 7)))
    sintra_panorama = build_sintra_panorama_sift(sintra1, sintra2, sintra3, sintra4, sintra5, True)
    sintra_panorama = build_sintra_panorama_manual(sintra1, sintra2, sintra3, sintra4, sintra5)
    # cv2.imwrite('panorama_sintra_manual_without_ransac.jpg', sintra_panorama)

    # # beach panorama
    beach1 = cv2.imread('data/beach1.jpg')
    beach2 = cv2.imread('data/beach2.jpg')
    beach3 = cv2.imread('data/beach3.jpg')
    beach4 = cv2.imread('data/beach4.jpg')
    beach5 = cv2.imread('data/beach5.jpg')
    beach1 = cv2.resize(beach1, (int(beach1.shape[1] / 6), int(beach1.shape[0] / 6)))
    beach2 = cv2.resize(beach2, (int(beach2.shape[1] / 6), int(beach2.shape[0] / 6)))
    beach3 = cv2.resize(beach3, (int(beach3.shape[1] / 6), int(beach3.shape[0] / 6)))
    beach4 = cv2.resize(beach4, (int(beach4.shape[1] / 6), int(beach4.shape[0] / 6)))
    beach5 = cv2.resize(beach5, (int(beach5.shape[1] / 6), int(beach5.shape[0] / 6)))
    beach_panorama = build_beach_panorama_sift(beach1, beach2, beach3, beach4, beach5, True)
    beach_panorama = build_beach_panorama_manual(beach1, beach2, beach3, beach4, beach5)
    cv2.imwrite('beach_panorama.jpg', beach_panorama)

    #2.9+2.10
    left = cv2.imread('my_data/our_panorama_left.jpeg')
    middle = cv2.imread('my_data/our_panorama_middle.jpeg')
    right = cv2.imread('my_data/our_panorama_right.jpeg')

    left = cv2.resize(left, (int(left.shape[1] / 4), int(left.shape[0] / 4)))
    middle = cv2.resize(middle, (int(middle.shape[1] / 4), int(middle.shape[0] / 4)))
    right  = cv2.resize(right, (int(right.shape[1] / 4), int(right.shape[0] / 4)))

    panorama = build_our_panorama_blender(left, middle, right, ransac=False)
    cv2.imwrite('our_panorama_blend.png',panorama)
    panorama = build_our_panorama(left, middle, right, ransac=False)
    cv2.imwrite('our_panorama.png',panorama)

    left = cv2.imread('my_data/panorama_technion/left.jpeg')
    middle = cv2.imread('my_data/panorama_technion/middle.jpeg')
    right = cv2.imread('my_data/panorama_technion/right.jpeg')

    left = cv2.resize(left, (int(left.shape[1] / 4), int(left.shape[0] / 4)))
    middle = cv2.resize(middle, (int(middle.shape[1] / 4), int(middle.shape[0] / 4)))
    right  = cv2.resize(right, (int(right.shape[1] / 4), int(right.shape[0] / 4)))

    panorama = build_our_panorama_blender(right, middle, left, ransac=False)
    cv2.imwrite('our_panorama_technion_blend.png',panorama)
    panorama = build_our_panorama(left, middle, right, ransac=True)
    cv2.imwrite('our_panorama_technion.png',panorama)

#%%
class StraightLines():

    def __init__(self, m, c):
        self.slope = m
        self.y_intercept = c

    def __call__(self, x):
        return self.slope * x + self.y_intercept


line = StraightLines(0.4, 3)

for x in range(-5, 6):
    print(x, line(x))
lines = []
lines.append(StraightLines(1, 0))
lines.append(StraightLines(0.5, 3))
lines.append(StraightLines(-1.4, 1.6))

import matplotlib.pyplot as plt
import numpy as np

# X = np.linspace(-5,5,100)

for index, line in enumerate(lines):
    y=[]
    for x in np.linspace(-5,5,100):
        y.append(line(x))
    plt.plot(X, y, label='line' + str(index))

plt.title('Some straight lines')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

#%%

# def my_Resize (*args) :
#     l=[]
#     for arg in args :
#         arg = cv2.resize(arg, (int(arg.shape[1] / 6), int(arg.shape[0] / 6)))
#         l.append(arg)
#     return l


def my_Resize_arr (*args) :
    arr = np.zeros((200,266,3))
    for i in range(len(args)) :
        arr[i] = cv2.resize(args[i], (int(args[i].shape[1] / 6), int(args[i].shape[0] /  6)))
    return arr
