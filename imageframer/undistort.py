import collections
import cPickle as pickle
import os
import sys

import cv2
import numpy as np


CalibrationSettings = collections.namedtuple('CalibrationSettings',
                                             ('camera_matrix', 'distort_coeffs',
                                              'optimal_camera',
                                              'area_threshold'))
_cached_profiles = dict()


def _profile_filename(profile_name):
    return os.path.join(os.path.dirname(__file__), profile_name+'.pickle')


def load_calibration_profile(profile_name):
    global _cached_profiles
    if profile_name not in _cached_profiles:
        with open(_profile_filename(profile_name), 'rb') as fin:
            _cached_profiles[profile_name] = pickle.load(fin)
    return _cached_profiles[profile_name]


def update_calibration_profile(profile_name, resolution, camera_matrix,
                               distort_coeffs, optimal_camera, area_threshold):
    try:
        profile = load_calibration_profile(profile_name)
    except IOError:
        profile = dict()

    settings = CalibrationSettings(camera_matrix=camera_matrix,
                                   distort_coeffs=distort_coeffs,
                                   optimal_camera=optimal_camera,
                                   area_threshold=area_threshold)
    profile[resolution] = settings

    with open(_profile_filename(profile_name), 'wb') as fout:
        pickle.dump(profile, fout, protocol=pickle.HIGHEST_PROTOCOL)

    _cached_profiles[profile_name] = profile


def _get_area(corners):
    # https://en.wikipedia.org/wiki/Quadrilateral#Non-trigonometric_formulas
    a, b, c, d = (np.linalg.norm(c)**2 for
                  c in (corners - np.roll(corners, 1, axis=0)))
    p = np.linalg.norm(corners[0]-corners[2])**2
    q = np.linalg.norm(corners[1]-corners[3])**2
    return 0.25 * np.sqrt(4*p*q - (a+c-b-d)**2)


def _uncrop(img, corners):
    '''Return the parent image of `img`, if `img` is a view.'''
    if img.base is None:
        # The image doesn't have a parent, so there's nothing to do.
        return img, corners

    base_slice = get_base_slice(img)
    img = img.base
    corners = corners.copy() + (base_slice[0].start, base_slice[1].start)

    return img, corners


def test_uncrop():
    from .main import crop_to_corners

    img = np.random.randint(0, 255, 1024*768).astype(np.uint8)
    img = img.reshape((768, 1024)).copy()
    corners = np.array([[10, 10], [768-20, 20], [768-30, 1024-30],
                        [40, 1024-40]], dtype=np.float)

    img2, corners2 = _uncrop(img, corners)
    if not np.all(img2 == img) or not np.all(corners2 == corners):
        raise RuntimeError('Uncropped non-cropped image failed.')

    cropped_img, cropped_corners = crop_to_corners(img, corners)
    img3, corners3 = _uncrop(cropped_img, cropped_corners)
    if not np.all(img3 == img) or not np.all(corners3 == corners):
        raise RuntimeError('Uncropped cropped image failed.')


def should_undistort(img, corners, profile_name):
    profile = load_calibration_profile(profile_name)
    img, corners = _uncrop(img, corners)
    if img.shape[:2] not in profile:
        sys.stderr.write('WARNING: No calibration data for resolution '
                         '{}\n'.format(img.shape[:2]))
        return False

    settings = profile[img.shape[:2]]
    area = _get_area(corners)
    return area >= settings.area_threshold


def undistort(img, corners, profile_name):
    profile = load_calibration_profile(profile_name)
    img, corners = _uncrop(img, corners)
    if len(img.shape) == 3:
        # XXX Hack. Fix _uncrop!
        img = img[:, :, 1]

    if img.shape[:2] not in profile:
        raise ValueError('No calibration settings for input image size.')

    settings = profile[img.shape[:2]]
    height, width = img.shape[:2]

    # Undistort corners. OpenCV expects (x, y) and a 3D array.
    corners = np.array([np.fliplr(corners)])
    undist_corners = cv2.undistortPoints(corners, settings.camera_matrix,
                                         settings.distort_coeffs, None, None,
                                         settings.optimal_camera)
    undist_corners = np.fliplr(undist_corners[0])

    undist_img = cv2.undistort(img, settings.camera_matrix,
                               settings.distort_coeffs, None,
                               settings.optimal_camera)

    return undist_img, undist_corners


def get_base_slice(view):
    '''Returns the 2D slice of view's parent that yields view.

    Note that this function will return a 2D slice even if view's parent
    is three-dimensional.'''
    if view.dtype != np.uint8:
        raise ValueError('Don''t know how to compute offset for types other '
                         'than uint8.')
    if not hasattr(view, 'base') or view.base is None:
        raise ValueError('`view` argument does not have `base` property.')
    if len(view.base.strides) not in (2, 3):
        raise ValueError('Base array must have two or three dimensions.')

    # Determine the strides
    base = view.base
    ystride, xstride = base.strides[:2]

    # Compute difference between base start and the view's end & start.
    base_start, _ = np.byte_bounds(base)
    view_start, view_end = np.byte_bounds(view)
    diff_start = view_start - base_start
    diff_end = view_end - base_start - 1

    # Compute slice offsets
    ystart = diff_start / ystride
    xstart = (diff_start % ystride) / xstride
    yend = diff_end / ystride
    xend = (diff_end % ystride) / xstride

    return slice(ystart, yend+1), slice(xstart, xend+1)


def test_get_base_slice():
    width = 21
    height = 13
    frame2 = np.zeros((height, width), dtype=np.uint8)
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    for xstart in xrange(width+1):
        for xend in xrange(xstart+1, width+1):
            for ystart in xrange(height+1):
                for yend in xrange(ystart+1, height+1):
                    cur_slice = slice(ystart, yend), slice(xstart, xend)

                    cropped2 = frame2[cur_slice]
                    if cur_slice != get_base_slice(cropped2):
                        raise ValueError('Failure at 2D slice ', cur_slice)

                    cropped3 = frame3[cur_slice + (1, )]
                    if cur_slice != get_base_slice(cropped3):
                        raise ValueError('Failure at 3D slice ', cur_slice)
