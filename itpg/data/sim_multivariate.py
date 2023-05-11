"""Multivariate data generators."""
# Part of the code is inspired from Frites python package.
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 01/2023

import numpy as np

# implemented bivariate relationships
BIV_REL = ['linear', 'mustache', 'rotated_plane',
           'smile', 'mirror', 'circle', 'clusters']

# --------------- Bivariate generators ---------------

# TODO: add xmin, xmax to functions


def generate_biv_normal(n, mean=0.0, cov=0.4):
    """Generate bivariate normals."""
    sd = np.array([[1, cov], [cov, 1]])
    mean = np.array([mean, mean])
    xy = np.random.multivariate_normal(mean, sd, size=n)
    # TODO: think this
    # xy += np.random.rand(*xy.shape)/1000.0
    x = xy[:, 0]
    y = xy[:, 1]
    return x, y


def generate_mustache(n):
    """Generate data with mustache shape."""
    x = np.linspace(-1, 1, n)
    r = 2*(np.random.random(n)) - 1
    y = 4.0*(x*x - 0.5)**2 + r/3
    return x, y


def generate_rotated_plane(n, f=4.0):
    """Generate rotated plane by a angle of -pi/f."""
    x = np.linspace(-1, 1, n)
    y = 2*(np.random.random(n)) - 1
    # translates slice objects to concatenation along the second axis
    xy = np.c_[x, y]
    # rotate plane
    t = -np.pi/f
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]).T
    xy = np.dot(xy, rot)
    # get x, y
    x, y = xy[:, 0], xy[:, 1]
    return x, y


def generate_smile(n):
    """Generate data with smile/convex shape."""
    x = np.linspace(-1, 1, n)
    r = 2*(np.random.random(n)) - 1
    y = 2*(x*x)+r
    return x, y


def generate_mirror(n):
    """Generate data with mirror (concave/convex) shape."""
    x = np.linspace(-1, 1, n)
    r = np.random.random(n)/2.0
    y = x*x+r
    # flip indices
    indices = np.random.permutation(len(y))[:int(n/2)]
    y[indices] = -y[indices]
    return x, y


def generate_circle(n):
    """Generate data with circle shape."""
    x = np.linspace(-1, 1, n)
    r = np.random.normal(0, 1/8.0, n)
    y = np.cos(x*np.pi)+r
    r = np.random.normal(0, 1 / 8.0, n)
    x = np.sin(x * np.pi) + r
    return x, y


def generate_clusters(n, mean=3):
    """Generate data with 4 clusters shape."""
    sd = np.array([[1, 0], [0, 1]])
    xy1 = np.random.multivariate_normal([mean, mean], sd, int(n / 4))
    xy2 = np.random.multivariate_normal([-mean, mean], sd, int(n / 4))
    xy3 = np.random.multivariate_normal([-mean, -mean], sd, int(n / 4))
    xy4 = np.random.multivariate_normal([mean, -mean], sd, int(n / 4))
    xy = np.r_[xy1, xy2, xy3, xy4]
    x, y = xy[:, 0], xy[:, 1]
    return x, y


def generate_data_biv(n, rel='linear', **kwargs):
    """Generate bivariate simulated data."""
    assert rel in BIV_REL

    if rel == 'linear':
        cov = kwargs['cov'] if 'cov' in kwargs.keys() else 0.4
        mean = kwargs['mean'] if 'mean' in kwargs.keys() else 0.0
        x, y = generate_biv_normal(n, mean, cov)
    elif rel == 'mustache':
        x, y = generate_mustache(n)
    elif rel == 'rotated_plane':
        f = kwargs['f'] if 'f' in kwargs.keys() else 4.0
        x, y = generate_rotated_plane(n, f)
    elif rel == 'smile':
        x, y = generate_smile(n)
    elif rel == 'mirror':
        x, y = generate_mirror(n)
    elif rel == 'circle':
        x, y = generate_circle(n)
    elif rel == 'clusters':
        mean = kwargs['mean'] if 'mean' in kwargs.keys() else 3.0
        x, y = generate_clusters(n, mean)

    return x, y
