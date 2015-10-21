#!/usr/bin/env python

import click
import numpy as np
import matplotlib.pyplot as plt

from .. import undistort


@click.command()
@click.argument('profile_name')
@click.argument('images', nargs=-1)
def main(profile_name, images):
    for img_name in images:
        img = plt.imread(img_name)
        undist, _ = undistort.undistort(img, np.zeros((4, 2)), profile_name)

        plt.figure()
        plt.imshow(img[:, :, 1], cmap=plt.cm.Greys_r, interpolation='nearest')
        plt.title('Original')
        plt.figure()
        plt.imshow(undist, cmap=plt.cm.Greys_r, interpolation='nearest')
        plt.title('After undistortion')
        plt.show()


if __name__ == '__main__':
    main.main()
