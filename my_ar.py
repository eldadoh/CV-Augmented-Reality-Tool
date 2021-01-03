from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
from PIL import Image
import os
import frame_video_convert
import copy
import my_homography as mh

def convert_PIL_to_cv2(PILimage):
    pil_image = PILimage.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def getPoints(im1, N):
    # plt.figure()
    # plt.imshow(im1)
    # p1 = plt.ginput(N, timeout=150)
    # plt.close()
    p1 = np.asarray(([286, 436], [700, 495], [611, 1147], [135, 1060]))
    p2 = np.asarray(([0, 0], [200, 0], [200, 320], [0, 320]))
    return np.asarray(p1).T, np.asarray(p2).T


# HW functions:
def create_ref(im_path):
    """
       Your code here
    """
    im1 = cv2.imread(im_path)

    im2 = np.zeros((320, 200, 3))
    p1, p2 = getPoints(im1, 4)
    H, status = cv2.findHomography(p1.T, p2.T)
    ref_image = cv2.warpPerspective(im1, H, (200, 320))
    return ref_image

def my_vid2vid(vid1_path,vid2_path):
    book_path = 'my_data/my_book.jpeg'
    ref_book = create_ref(book_path)

    vid1_cap = cv2.VideoCapture(vid1_path)
    vid2_cap = cv2.VideoCapture(vid2_path)
    success1, image1 = vid1_cap.read()
    success2, image2 = vid2_cap.read()
    count = 0
    print("converting video to frames...")
    while success1 and success2:
        fname = str(count).zfill(4)
        im2im_result = im2im(image1, ref_book, image2)
        cv2_image = convert_PIL_to_cv2(im2im_result)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('my_data/frames_for_video_new/', fname + ".png"), cv2_image)
        success1, image1 = vid1_cap.read()
        success2, image2 = vid2_cap.read()
        # print('Read a new frame: ', success)
        count += 1

def im2im(scene_image, ref_book, object_image):
    ref_book_points = np.asarray(([0, 0], [0, 200], [320, 200], [320, 0]))
    p1, p2 = mh.getPoints_SIFT(scene_image, ref_book)
    H_scene2ref, status = cv2.findHomography(p2.T, p1.T, cv2.RANSAC)
    object_corners = np.asarray(([0, 0], [object_image.shape[1], 0], [object_image.shape[1], object_image.shape[0]], [0, object_image.shape[0]]))
    i=0
    points_in_scene = np.zeros((2, 4))
    for p in ref_book_points:
        x, y, z = np.dot(H_scene2ref, np.array([[p[1]], [p[0]], [1]]))
        points_in_scene[:,i] = np.asarray([int(x / z), int(y / z)]).T
        i+=1
    h_object2scene, status = cv2.findHomography(object_corners, points_in_scene.T)
    wrap_object = cv2.warpPerspective(object_image, h_object2scene, (scene_image.shape[1], scene_image.shape[0]))
    # image = mh.imageStitching(wrap_cv_book, scene_image, (0,0))
    scene_image = Image.fromarray(scene_image)
    mask = cv2.cvtColor(wrap_object, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255
    wrap_object = Image.fromarray(wrap_object)
    mask = Image.fromarray(mask)
    image = Image.composite(wrap_object,scene_image , mask)
    return image

if __name__ == '__main__':
    print('my_ar')
    # 3.1
    ref_book_points = np.asarray(([0, 0], [0, 200], [320, 200], [320, 0]))
    book_path = 'my_data/my_book.jpeg'
    ref_book = create_ref(book_path)
    # 3.2
    book2_path = cv2.imread('my_data/my_book_another_scene.jpeg')
    cv_book = cv2.imread('my_data/cv_book.JPG')
    delipekan_image = cv2.imread('my_data/delipekan.jpeg')
    loah_image = cv2.imread('my_data/loah.jpeg')
    scence_with_cvbook = im2im(book2_path, ref_book, cv_book)
    scence_with_delipekan = im2im(book2_path, ref_book, delipekan_image)
    scence_with_loah = im2im(book2_path, ref_book, loah_image)
    scence_with_cvbook.save('scence_with_cvbook.jpg')
    scence_with_delipekan.save('scence_with_delipekan.jpg')
    scence_with_loah.save('scence_with_loah.jpg')

    3.3
    vid1_path= 'my_data/3_3_new.mp4'
    vid2_path = 'my_data/video_input.avi'
    my_vid2vid(vid1_path, vid2_path)
    frame_video_convert.image_seq_to_video('my_data/frames_for_video_new', output_path='./video.mp4', fps=15.0)






