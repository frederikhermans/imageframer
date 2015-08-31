import cv2
import numpy as np


def find_circles(img):
    """Return contour objects for all circles in `img`"""
    _, biimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(biimg, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    circles = [Contour(c) for c in contours]
    return [c for c in circles if c.is_circle()]


class Contour(object):
    """A class representing a contour.

    The main purpose is to cache repeated computations of properties such
    as area, or bounding rectangle."""

    def __init__(self, contour):
        self.contour = contour
        self._area = None
        self._boundingRect = None
        self._perimeter = None
        self.moment = None

    def is_circle(self):
        # https://opencv-code.com/tutorials/detecting-simple-shapes-in-an-image/
        # XXX The parameters below are very much tweaked to our
        # phase-error-new dataset.
        if len(self.contour) < 6:
            return False

        # Reject very large circles, and contours that
        # are not as wide as they are high
        _, _, width, height = self.boundingRect
        width, height = float(width), float(height)
        if width > 200 or height > 200 or \
           np.abs(1. - width/height) > .2:
            return False

        # Reject shapes where the area is very much unlike
        # radius*pi^2
        area = self.area
        if area < 15:
            return False
        radius = width/2.
        if np.abs(1. - (area / (np.pi*radius**2))) > .5:
            return False

        if np.abs(1. - (self.perimeter / (2*np.pi*radius))) > .25:
            return False

        return True

    @property
    def area(self):
        if self._area is None:
            self._area = cv2.contourArea(self.contour)
        return self._area

    @property
    def boundingRect(self):
        if self._boundingRect is None:
            self._boundingRect = cv2.boundingRect(self.contour)
        return self._boundingRect

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = cv2.arcLength(self.contour, True)
        return self._perimeter

    @property
    def centroid_x(self):
        if self.moment is None:
            self.moment = cv2.moments(self.contour)
            self._centroid_x = self.moment['m10']/self.moment['m00']
            self._centroid_y = self.moment['m01']/self.moment['m00']
        return self._centroid_x

    @property
    def centroid_y(self):
        if self.moment is None:
            self.moment = cv2.moments(self.contour)
            self._centroid_x = self.moment['m10']/self.moment['m00']
            self._centroid_y = self.moment['m01']/self.moment['m00']
        return self._centroid_y

    def __repr__(self):
        return '({:.2f}, {:.2f})'.format(self.centroid_y, self.centroid_x)
