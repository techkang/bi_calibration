# -*- coding:utf-8 -*-
# __author__ = "Kang Sheng"
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import fire


class Config(object):
    sample = '01'  # 测试图片
    disp_calib = False  # 是否展示单目校正结果
    stereo_calib = True  # 是否进行双目校正
    disp_stereo_calib = False  # 是否展示双目校正结果
    disparity = True  # 是否利用视差估算距离
    R = dict({'01': np.array([[1, -0.0032, -0.005], [0.0033, 0.9999, 0.0096],
                              [0.0057, -0.0097, 0.9999]])})  # 由 MATLAB 标定的旋转矩阵
    T = dict({'01': np.array([-83.0973, 1.0605, 0.0392])})  # 由 MATLAB 标定的平移矩阵
    # TODO: 补充其他图像的R和T
    matlab = True  # 在双目校正时是否使用 matlab 标定的值
    num = 3  # StereoSGBM_create 函数参数：最小可能的差异值
    blockSize = 5  # StereoSGBM_create 函数参数：匹配的块大小。


opt = Config()


def calibration(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../left/*.jpg')
    images_r = glob.glob('../right/*.jpg')
    images.sort()
    images_r.sort()

    for fname, fname_r in zip(images, images_r):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_r = cv2.imread(fname_r)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True and ret_r == True:
            objpoints.append(objp)
            objpoints_r.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                          criteria)
            imgpoints.append(corners2)
            imgpoints_r.append(corners2_r)

            # Draw and display the corners
            if opt.disp_calib:
                cv2.imshow('img', img)
                cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)
    img = cv2.imread('../left/left' + str(opt.sample) + '.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    if opt.disp_calib:
        cv2.imwrite('../calibresult/left' + str(opt.sample) + '.png', dst)

    ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(objpoints_r,
                                                           imgpoints_r,
                                                           gray_r.shape[::-1],
                                                           None, None)
    img_r = cv2.imread('../right/right' + str(opt.sample) + '.jpg')
    h, w = img_r.shape[:2]
    newcameramtx_r, roi = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w, h),
                                                        1, (w, h))
    # undistort
    dst_r = cv2.undistort(img_r, mtx_r, dist_r, None, newcameramtx_r)

    # crop the image
    x, y, w, h = roi
    dst_r = dst_r[y:y + h, x:x + w]
    if opt.disp_calib:
        cv2.imwrite('../calibresult/right' + str(opt.sample) + '.png', dst)

    if not opt.stereo_calib:
        exit(0)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, mtx,
                            dist, mtx_r, dist_r, gray.shape[::-1])

    if opt.matlab:
        try:
            R = opt.R[opt.sample]
            T = opt.T[opt.sample]
        except:
            print('Please modify config to add R and T for ' + opt.sample)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        gray.shape[::-1], R, T)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                       distCoeffs1, R1, P1,
                                                       gray.shape[::-1],
                                                       cv2.INTER_NEAREST)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                         distCoeffs2, R2,
                                                         P2, gray.shape[::-1],
                                                         cv2.INTER_NEAREST)

    img = cv2.imread('../left/left' + str(opt.sample) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(('../right/right' + str(opt.sample) + '.jpg'))
    gray_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgL = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
    imgR = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)

    if opt.disp_stereo_calib:
        cv2.imwrite(
            '../result/stereo_calibresult/left' + str(opt.sample) + '.png',
            imgL)
        cv2.imwrite(
            '../result/stereo_calibresult/right' + str(opt.sample) + '.png',
            imgR)

        plt.subplot(121)
        plt.title('left')
        plt.imshow(imgL, cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.title('right')
        plt.imshow(imgR, cmap='gray')
        plt.axis('off')
        plt.show()

    if not opt.disparity:
        exit(0)

    cv2.namedWindow("depth")
    cv2.namedWindow("disparity")
    cv2.moveWindow("depth", 0, 0)
    cv2.moveWindow("disparity", 600, 0)

    def callbackFunc(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print(threeD[y][x])

    cv2.setMouseCallback("depth", callbackFunc, None)

    stereo = cv2.StereoSGBM_create(numDisparities=16 * opt.num,
                                   blockSize=opt.blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)

    cv2.imshow("disparity", disp)
    cv2.imshow("depth", imgL)

    key = cv2.waitKey(0)
    if key == ord("q"):
        exit(0)
    elif key == ord("s"):
        cv2.imwrite("../result/disparity/disparity"+opt.sample+".png", disp)


if __name__ == '__main__':
    fire.Fire(calibration)
