#!/usr/bin/env python

# Adapted from
# https://github.com/Itseez/opencv/blob/master/samples/python2/calibrate.py

# Python 2/3 compatibility
from __future__ import print_function

import os

import numpy as np
import cv2

from .. import undistort


USAGE = ('USAGE: calib.py [--save <profilename>] [--debug <output path>] '
         '[--square_size] [--treshold <pixels>] [<image mask>]')


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '',
                                   ['save=', 'debug=', 'square_size=',
                                    'threshold='])
    args = dict(args)
    try:
        img_mask = img_mask[0]
    except:
        img_mask = '../data/left*.jpg'

    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    square_size = float(args.get('--square_size', 1.0))
    threshold = args.get('--threshold', None)
    profile_name = args.get('--save', None)
    if profile_name and threshold is None:
        print('Cannot --save without --threshold.')
        sys.exit(-1)

    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    for fn in img_names:
        print('processing %s...' % fn,)
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            cv2.imwrite('%s/%s_chess.bmp' % (debug_dir, name), vis)
        if not found:
            print('chessboard not found')
            continue
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print('ok')

    rms, camera_matrix, dist_coefs, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    dist_coefs = dist_coefs.ravel()
    newcamera, _ =\
        cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    cv2.destroyAllWindows()

    if profile_name:
        undistort.update_calibration_profile(profile_name,
                                             resolution=(h, w),
                                             camera_matrix=camera_matrix,
                                             optimal_camera=newcamera,
                                             distort_coeffs=dist_coefs,
                                             area_threshold=int(threshold))
