import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import cluster
from . import contour
from . import linetools


def create_marker(size):
    """Returns a circular marker."""
    if size % 2:
        raise('Marker size must be a multiple of 2.')
    radius = int(round(0.4 * size))
    marker = np.ones((size, size), dtype=np.uint8)*255
    cv2.circle(marker, (size/2, size/2), radius, 0, -1)
    return marker


class Framer(object):
    def __init__(self, frame_shape, border_size):
        if frame_shape[0] != frame_shape[1]:
            raise ValueError('Frame must be square.')

        self.border_pixels = int(frame_shape[0] * border_size)
        if self.border_pixels % 2:
            self.border_pixels += 1
        self.frame_shape = frame_shape
        self._init_template()

    def _init_template(self, draw_cross=False):
        """Construct a template for adding markers to a frame."""
        frame_height, frame_width = self.frame_shape

        circle = create_marker(self.border_pixels)
        circle_size = self.border_pixels

        # Construct the left border first
        left = np.ones((frame_height, circle_size), dtype=np.uint8)*255
        offset = (frame_height % circle_size) / 2
        self.circles_per_side = 0
        for i in xrange(frame_height/circle_size):
            if i != 1 and i != 3:
                left[offset:offset+circle_size, :] = circle
                self.circles_per_side += 1
            offset += circle_size

        if self.circles_per_side < 2:
            # We need at least 2 markers per side
            raise ValueError('Border size too large.')
        print self.circles_per_side

        # `fill` will be pasted into the corners.
        fill = np.ones((self.border_pixels, self.border_pixels), dtype=np.uint8)
        fill *= 255
        # Add a small cross
        if draw_cross:
            c = left.shape[1]/2
            fill[c-5:c+6, c] = 0
            fill[c, c-5:c+6] = 0

        # Construct the actual template
        frame = np.zeros(self.frame_shape, np.uint8)
        center = np.hstack((left, frame, np.flipud(left)))
        top = np.hstack((fill, np.rot90(left), fill))
        self.template = np.vstack((top, center, np.fliplr(top)))

        # Corners of the template
        corners = np.array(self.template.shape)*((0, 0), (1, 0), (1, 1), (0, 1))
        factors = np.array(((1, 1), (-1, 1), (-1, -1), (1, -1)))
        # Corners of the frame
        self.frame_corners = corners + self.border_pixels*factors
        # Intersection points
        self.intersections = corners + self.border_pixels*0.5*factors
        # Need to subtract 1 from the y-coordinates, because np.rot90 will
        # shift the centroid of the markers, since our markers are not
        # perfectly symmetric.
        self.intersections[:, 0] -= 1

    def add_markers(self, frame):
        """Returns `frame` padded with markers."""
        if frame.dtype != np.uint8:
            raise ValueError('Can only operate on uint8.')
        marked_frame = self.template.copy()
        frame_slice = (slice(self.border_pixels, -self.border_pixels),
                       slice(self.border_pixels, -self.border_pixels))
        marked_frame[frame_slice] = frame
        return marked_frame

    def locate(self, img, debug=()):
        """Returns the corners of a frame in `img`."""
        if img.dtype != np.uint8:
            raise ValueError('Can only operate on uint8 data.')

        if isinstance(debug, basestring):
            debug = (debug, )

        circles = contour.find_circles(img)

        # Cluster circles by their area
        area_clusters = cluster.compute_clusters(circles, 'area')
        if len(area_clusters) == 0:
            raise ValueError('No circles found.')

        # Sort clusters by whether they have the right number of elements, and
        # also by the area of the elements
        def sort_key(c):
            return (abs(4*self.circles_per_side - len(c)), -c[0].area)
        area_clusters.sort(key=sort_key)
        # This is the best cluster
        markers = area_clusters[0]

        if 'markers' in debug:
            # Plot all detected circles.
            extent = (0, img.shape[1], 0, img.shape[0])
            plt.figure()
            for i, contours in enumerate(area_clusters):
                color = 'r' if i == 0 else 'b'
                plt.imshow(_draw_contours(contours, img.shape, color),
                           extent=extent, interpolation='nearest')
            plt.imshow(img, cmap=plt.cm.Greys_r, extent=extent, alpha=0.4,
                       interpolation='nearest')

        # Cluster markers by their location
        top, left, bottom, right = cluster.group_markers(markers,
                                                         self.circles_per_side)
        # Fit a line through each set of markers
        lines = list()
        for ms in (top, left, bottom, right):
            centers = np.array([(m.centroid_y, m.centroid_x) for m in ms])
            lines.append(linetools.fit_line(centers))

        # Compute intersections of the lines through the markers
        intersections = list()
        for line_a, line_b in zip(lines, lines[1:] + [lines[0]]):
            intersections.append(linetools.intersect(line_a, line_b))
        intersections = np.array(intersections)

        # Create a transform from the template intersections to the
        # intersections in the input image.
        m = cv2.getPerspectiveTransform(self.intersections.astype(np.float32),
                                        intersections.astype(np.float32))

        frame_corners = self.frame_corners.astype(np.float32).reshape(1, 4, 2)
        corners = cv2.perspectiveTransform(frame_corners, m)[0]

        if 'result' in debug:
            _plot_result(img, top, left, bottom, right, intersections, corners)

        return corners

    def extract(self, img, output_shape, corners=None):
        """Extract a frame from `img` and correct perspective distortion."""
        if img.dtype != np.uint8:
            raise ValueError('Can only operate on uint8.')
        if corners is None:
            corners = self.locate(img)

        corners = np.fliplr(corners).astype(np.float32)
        dst_corners = np.array(output_shape) * ((0, 0), (1, 0), (1, 1), (0, 1))
        dst_corners = np.fliplr(dst_corners).astype(np.float32)
        m = cv2.getPerspectiveTransform(corners, dst_corners)

        return cv2.warpPerspective(img, m, output_shape)


def _draw_contours(contours, shape, color):
    conv = matplotlib.colors.ColorConverter()
    img = np.zeros(list(shape) + [4])
    color = conv.to_rgba(color)
    cv2.drawContours(img, [c.contour for c in contours], -1, color, -1)
    return img


def _plot_corners(points, **kwargs):
    xs = list(points[:, 1])
    ys = list(points[:, 0])
    plt.plot(xs + [xs[0]], ys + [ys[0]], **kwargs)


def _plot_result(img, top, left, bottom, right, intersections, corners):
    plt.figure()
    extent = (0, img.shape[1], img.shape[0], 0)
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.Greys_r, extent=extent)
    axis = plt.axis()

    for color, markers in zip('gycm', (top, left, bottom, right)):
        plt.imshow(_draw_contours(markers, img.shape, color), alpha=0.4,
                   interpolation='nearest', extent=extent)
        plt.scatter([m.centroid_x for m in markers],
                    [m.centroid_y for m in markers], color='r')

    _plot_corners(intersections, color='r', marker='x')
    _plot_corners(corners, color='r', linestyle='--', marker='+')

    plt.axis(axis)
