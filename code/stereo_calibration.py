import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import fire


def calibration(des='12'):
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

    # pattern_size=(7, 6)
    # img_left_points=[]
    # img_right_points=[]
    # obj_points=[]
    # pattern_points=objp
    #
    # for img,img_r in zip(images,images_r):
    #
    #     left_img = cv2.imread(img, cv2.CV_8UC1)
    #     right_img = cv2.imread(img_r, cv2.CV_8UC1)
    #
    #     image_size = left_img.shape
    #
    #     find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    #
    #     left_found, left_corners = cv2.findChessboardCorners(left_img, pattern_size,
    #                                                          flags=find_chessboard_flags)
    #     right_found, right_corners = cv2.findChessboardCorners(right_img, pattern_size,
    #                                                            flags=find_chessboard_flags)
    #
    #     if left_found:
    #         cv2.cornerSubPix(left_img, left_corners, (11, 11), (-1, -1), (
    #         cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    #     if right_found:
    #         cv2.cornerSubPix(right_img, right_corners, (11, 11), (-1, -1), (
    #         cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    #
    #     if left_found and right_found:
    #         img_left_points.append(left_corners)
    #         img_right_points.append(right_corners)
    #         obj_points.append(pattern_points)
    #
    #     cv2.imshow("left", left_img)
    #     cv2.drawChessboardCorners(left_img, pattern_size, left_corners,
    #                               left_found)
    #     cv2.drawChessboardCorners(right_img, pattern_size, right_corners,
    #                               right_found)
    #
    #     cv2.imshow("left chess", left_img)
    #     cv2.imshow("right chess", right_img)
    #
    # stereocalib_criteria = (
    # cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    # stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST \
    #                     | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL \
    #                     | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    #
    #
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_left_points, (640,480), None, None)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 1, (640, 480))
    #
    # ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_right_points, (640,480), None, None)
    # newcameramtx_r, roi = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (640, 480), 1, (640, 480))
    #
    #
    # stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F \
    #     = cv2.stereoCalibrate(obj_points, img_left_points, img_right_points, newcameramtx,
    #                           dist,newcameramtx_r,dist_r,image_size,
    #                           criteria=stereocalib_criteria, flags=stereocalib_flags)
    #
    # rectify_scale = 0  # 0=full crop, 1=no crop
    # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1,
    #                                                   distCoeffs1,
    #                                                   cameraMatrix2,
    #                                                   distCoeffs2,
    #                                                   (640, 480), R,
    #                                                   T,
    #                                                   alpha=rectify_scale)
    # left_maps = cv2.initUndistortRectifyMap(cameraMatrix1,
    #                                         distCoeffs1, R1, P1,
    #                                         (640, 480), cv2.CV_16SC2)
    # right_maps = cv2.initUndistortRectifyMap(cameraMatrix2,
    #                                          distCoeffs2, R2, P2,
    #                                          (640, 480), cv2.CV_16SC2)
    #
    #
    # img = cv2.imread('../left/left' + str(des) + '.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #
    # img = cv2.imread(('../right/right' + str(des) + '.jpg'))
    # gray_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # left_img_remap = cv2.remap(gray, left_maps[0], left_maps[1],
    #                            cv2.INTER_LANCZOS4)
    # right_img_remap = cv2.remap(gray_r, right_maps[0],
    #                             right_maps[1], cv2.INTER_LANCZOS4)
    #
    # plt.subplot(121)
    # plt.title('left')
    # plt.imshow(left_img_remap, cmap='gray')
    # plt.axis('off')
    # plt.subplot(122)
    # plt.title('right')
    # plt.imshow(right_img_remap, cmap='gray')
    # plt.axis('off')
    # plt.show()

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
    cv2.imwrite('../result/stereo_calibresult/left' + str(des) + '.png', dst)

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
    cv2.imwrite('../result/stereo_calibresult/right' + str(des) + '.png', dst_r)

    newcameramtx=mtx.copy()
    newcameramtx_r=mtx_r.copy()


    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, newcameramtx,
                            dist, newcameramtx_r, dist_r, gray.shape[::-1])

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

    img = cv2.imread('../left/left' + str(des) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(('../right/right' + str(des) + '.jpg'))
    gray_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    des_l = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
    des_r = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)

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
