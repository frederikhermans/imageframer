import itertools

import numpy as np


def compute_clusters(elems, attr, thresh=0.3):
    """Cluster the elements in `elems` on similarity of `e.attr`"""
    means = list()
    clusters = list()

    for cur in elems:
        # Try to add `cur' to a cluster, or create a new cluster
        # if no matching cluster is found
        cur_prop = getattr(cur, attr)
        for i in xrange(len(clusters)):
            cluster_i_len = len(clusters[i])
            if np.abs(1.0 - (cur_prop/means[i])) <= thresh:
                means[i] = (means[i]*cluster_i_len+cur_prop)/(cluster_i_len+1.)
                clusters[i].append(cur)
                break
        else:
            means.append(cur_prop)
            clusters.append([cur])

    return clusters


def group_markers(markers, circles_per_side):
    """Find groups of markers on the top, left, bottom, and right.

    XXX Would be nicer to implement this with a Hough transform
    to find maximal sets of collinear markers."""
    def xrel(a, b):
        return np.abs(a.centroid_x - b.centroid_x) < np.sqrt(a.area/(2*np.pi))

    def yrel(a, b):
        return np.abs(a.centroid_y - b.centroid_y) < np.sqrt(a.area/(2*np.pi))

    xclusters = _equivalence_classes(markers, xrel)
    xclusters = [cluster for cluster in xclusters if len(cluster) > 1]
    xclusters.sort(key=lambda cluster: cluster[0].centroid_x)
    if len(xclusters) < 2:
        raise ValueError('Not enough clusters found.')
    left = xclusters[0]
    right = xclusters[-1]

    yclusters = _equivalence_classes(markers, yrel)
    yclusters = [cluster for cluster in yclusters if len(cluster) > 1]
    yclusters.sort(key=lambda cluster: cluster[0].centroid_y)
    if len(yclusters) < 2:
        raise ValueError('Not enough clusters found.')
    top = yclusters[0]
    bottom = yclusters[-1]

    return top, left, bottom, right


def _equivalence_classes(markers, rel_fn):
    """Compute the equivalence classes of markers for a given relation."""
    # XXX Clean this code up, it's probably unnecessary slow
    rel = list()
    for a, b in itertools.combinations(markers, r=2):
        if rel_fn(a, b):
            rel.append((a, b))

    clusters = list()
    stack = list(markers)
    while len(stack) > 0:
        cluster = [stack.pop()]

        i = 0
        while i < len(cluster):
            a = cluster[i]
            i += 1
            for x, y in rel:
                if a == x and y not in cluster:
                    stack.remove(y)
                    cluster.append(y)
                elif a == y and x not in cluster:
                    stack.remove(x)
                    cluster.append(x)
        clusters.append(cluster)

    return clusters
