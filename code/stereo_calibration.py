import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import fire


def calibration(des='01',data_file='../stat/matlab.mat'):
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
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)
    img = cv2.imread('../left/left' + str(des) + '.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(objpoints_r,
                                                           imgpoints_r,
                                                           gray_r.shape[::-1],
                                                           None, None)
    img_r = cv2.imread('../right/right' + str(des) + '.jpg')
    h, w = img_r.shape[:2]
    newcameramtx_r, roi = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w, h),
                                                        1, (w, h))
    # undistort
    dst_r = cv2.undistort(img_r, mtx_r, dist_r, None, newcameramtx_r)

    # crop the image
    x, y, w, h = roi
    dst_r = dst_r[y:y + h, x:x + w]

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, mtx,
                            dist, mtx_r, dist_r, gray.shape[::-1])


    R=np.array([[1,-0.0032,-0.005],[0.0033,0.9999,0.0096],[0.0057,-0.0097,0.9999]])
    T=np.array([-83.0973,1.0605,0.0392])
    # TODO: import mat and read stat from mat file.

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        gray.shape[::-1], R, T)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                       distCoeffs1,R1, P1,
                                                       gray.shape[::-1],
                                                       cv2.INTER_NEAREST)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                         distCoeffs2, R2,
                                                         P2, gray.shape[::-1],
                                                         cv2.INTER_NEAREST)

    img = cv2.imread('../left/left' + str(des) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(('../right/right' + str(des) + '.jpg'))
    gray_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    des_l = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
    cv2.imwrite('../result/stereo_calibresult/left' + str(des) + '.png', des_l)
    des_r = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)
    cv2.imwrite('../result/stereo_calibresult/right' + str(des) + '.png', des_r)

    plt.subplot(121)
    plt.title('left')
    plt.imshow(des_l, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('right')
    plt.imshow(des_r, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    fire.Fire(calibration)
