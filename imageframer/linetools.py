import numpy as np


def to_normal(m, c):
    """Convert a line from the form mx+c=y to normal form."""
    # y=m*x is the same line moved to the origin
    # add theta because we want the angle that is *perpendicular*
    theta = np.angle(np.complex(1.0, m)) + np.pi/2.

    # The signed (!) distance between the moved line and the input
    # line is given by:
    rho = c/np.sqrt(m**2+1)

    return theta, rho


def fit_line(points):
    """Use linear regression to fit a line through a set of points."""
    xs = points[:, 1]
    ys = points[:, 0]

    A = np.vstack([xs, np.ones(len(xs))]).T
    (m, c), residuals, _, _ = np.linalg.lstsq(A, ys)
    theta, rho = to_normal(m, c)
    if len(residuals) == 0:
        residuals = [9999999]

    if residuals > 1.0:
        # The residuals of our line fitting may be very large when we're
        # fitting an (almost) vertical line. In that case, we try rotating
        # the points and fit a line through the rotated points. If the
        # new residuals are smaller, then we return the new line after
        # rotating it back.
        A = np.vstack([-np.array(ys), np.ones(len(ys))]).T
        (mrot, crot), residualsrot, _, _ = np.linalg.lstsq(A, xs)
        if residualsrot < residuals:
            theta, rho = to_normal(mrot, crot)
            theta -= np.pi/2.

    return theta, rho


def intersect(line1, line2):
    """Calculate intersection of two lines in normal form."""

    if parallel(line1, line2):
        raise Exception('Error: Lines (almost) parallel.')

    # http://www.gamedev.net/topic/516334-line-intersection-normal-form/#entry4357860
    theta1, rho1 = line1
    theta2, rho2 = line2

    x = (rho2 * np.sin(theta1) - rho1 * np.sin(theta2)) / np.sin(theta1-theta2)
    y = (rho2 * np.cos(theta1) - rho1 * np.cos(theta2)) / np.sin(theta2-theta1)

    return y, x


def parallel(line1, line2):
    """Returns whether two lines in normal form are parallel to each other."""
    EPS = 0.0000001
    theta1 = line1[0]
    theta2 = line2[0]
    return abs(np.sin(theta1-theta2)) < EPS or abs(np.sin(theta2-theta1)) < EPS
