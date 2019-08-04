import numpy as np
import cv2
import glob

def getCamera():
    lcamera = cv2.VideoCapture(2)
    rcamera = cv2.VideoCapture(1)

    return lcamera, rcamera

def getImageFromCamera(lvideo, rvideo):
    lcheck, lframe = lvideo.read()
    rcheck, rframe = rvideo.read()

    return lcheck, rcheck, lframe, rframe

def cameraCalibration(filePath):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_l = []  # 2d points in image plane.
    imgpoints_r = []  # 2d points in image plane.

    images_right = glob.glob(filePath[0] + "\*.jpg")
    images_left = glob.glob(filePath[1] + "\*.jpg")
    images_left.sort()
    images_right.sort()

    for i, fname in enumerate(images_right):
        img_l = cv2.imread(images_left[i])
        img_r = cv2.imread(images_right[i])

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

        # If found, add object points, image points (after refining them)
        objpoints.append(objp)

        if ret_l is True:
            rt1 = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                   (-1, -1), criteria)
            imgpoints_l.append(corners_l)

        if ret_r is True:
            rt2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                   (-1, -1), criteria)
            imgpoints_r.append(corners_r)

        img_shape = gray_l.shape[::-1]

    rt1, M1, d1, r1, t1 = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_shape, None, None)
    rt2, M2, d2, r2, t2 = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_shape, None, None)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5

    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l,
        imgpoints_r, M1, d1, M2,
        d2, img_shape,
        criteria=stereocalib_criteria, flags=flags)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, d1, M2, d2, img_shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
                                                      alpha=-1, newImageSize=img_shape)

    print('Intrinsic Matrix 1 (first camera matrix)')
    print(M1)
    print('-------------------------------------------------------------------\n')
    print('dist 1 (vector of distortion coefficients for first camera)')
    print(d1)
    print('-------------------------------------------------------------------\n')
    print('Intrinsic Matrix 2 (second camera matrix)')
    print(M2)
    print('-------------------------------------------------------------------\n')
    print('dist 2 (vector of distortion coefficients for second camera)')
    print(d2)
    print('-------------------------------------------------------------------\n')
    print('R (rotation matrix)')
    print(R)
    print('-------------------------------------------------------------------\n')
    print('T (translation vector)')
    print(T)
    print('-------------------------------------------------------------------\n')
    print('E (essential matrix)')
    print(E)
    print('-------------------------------------------------------------------\n')
    print('F (fundamental matrix)')
    print(F)
    print('-------------------------------------------------------------------\n')
    print('Projection Metrix Left')
    print(P1)
    print('-------------------------------------------------------------------\n')
    print('Projection Metrix Right')
    print(P2)
    print('-------------------------------------------------------------------\n')
    print('END')
    print('-------------------------------------------------------------------\n')
    print("Mean reprojection error: ", ret)

    return  P1, P2

def run(leftcamera, rightcamera):

    iCount = 0
    filePath = [None] * 2
    filePath[0] = "image\camera calibration\webcam\Left"
    filePath[1] = "image\camera calibration\webcam\Right"

    while True:

        _, _, lframe, rframe = getImageFromCamera(leftcamera, rightcamera)
        _, _, lframe_con, rframe_con = getImageFromCamera(leftcamera, rightcamera)

        cv2.imshow("Left", lframe)
        cv2.imshow("Right", rframe)

        gray_l = cv2.cvtColor(lframe_con, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rframe_con, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

        if ret_l is True:

            # Draw and display the corners
            cv2.drawChessboardCorners(lframe_con, (9, 6), corners_l, ret_l)
            cv2.imshow("Left", lframe_con)

        if ret_r is True:

            # Draw and display the corners
            cv2.drawChessboardCorners(rframe_con, (9, 6), corners_r, ret_r)
            cv2.imshow("Right", rframe_con)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            iCount = iCount +1
            cv2.imwrite(filePath[0] + "\Left " + str(iCount) + ".jpg", lframe)
            cv2.imwrite(filePath[1] + "\Right " + str(iCount) + ".jpg", rframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # closewebcam
    leftcamera.release()
    rightcamera.release()
    cv2.destroyAllWindows()
    projectionMetrixL, projectionMetrixR = cameraCalibration(filePath)

def main():
    # get camera object and check l&r camera is opened
    lcamera, rcamera = getCamera()
    if (lcamera.isOpened() & rcamera.isOpened()):
        run(lcamera, rcamera)

if __name__ == '__main__':
    main()