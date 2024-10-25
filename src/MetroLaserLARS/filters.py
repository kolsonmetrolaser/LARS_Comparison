# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:48:12 2024

@author: KOlson
"""
# External imports
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from numpy.typing import ArrayLike, NDArray
import scipy.signal as sig


def sgf(a: ArrayLike, n: int = 1, w: int = 101, p: int = 0) -> NDArray:
    """
    Recursively applies `scipy.signal.savgol_filter` `n` times to an array `a` with
    the settings `window_length = w` and `polyorder = p`

    Parameters
    ----------
    a : ArrayLike
        The data to be filtered. If x is not a single or double precision floating point array,
        it will be converted to type `numpy.float64` before filtering.
    n : int, optional
        The number of times to recursively apply the filter. The default is 1.
    w : int, optional
        The length of the filter window (i.e., the number of coefficients).
        `w` must be less than or equal to the size of `a`. If `w` is even, `1` is added automatically.
        The default is 101.
    p : int, optional
        The order of the polynomial used to fit the samples. `p` must be less than `w`. The default is 0.

    Returns
    -------
    NDArray
        The filtered data.
    """
    if w % 2 == 0:
        w += 1
    if n <= 0:
        return a
    elif n == 1:
        return sig.savgol_filter(a, w, p, mode='nearest')
    else:
        return sgf(sig.savgol_filter(a, w, p, mode='nearest'), n-1)


def whittaker_smooth(x, w, lam, differences=1):
    '''
    Penalized least squares algorithm for background fitting.
    See `airpls` for copyright and source information.

    input
        x:
            input data (i.e. chromatogram of spectrum)
        w:
            binary masks (value of the mask is zero if a point belongs to peaks
            and one otherwise)
        lam:
            parameter that can be adjusted by user. The larger lambda is,  the
            smoother the resulting background
        differences:
            integer indicating the order of the difference of penalties

    output:
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
#    D = csc_matrix(np.diff(np.eye(m), differences))
    D = sparse.eye(m, format='csc')
    for i in range(differences):
        D = D[1:] - D[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = sparse.diags(w, 0, shape=(m, m))
    A = sparse.csc_matrix(W + (lam * D.T * D))
    B = sparse.csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airpls(x, lam=100, porder=1, itermax=100):
    '''
    airpls.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
    Baseline correction using adaptive iteratively reweighted penalized least squares

    This program is a translation in python of the R source code of airPLS version 2.0
    by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls
    Reference:
    Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive
    iteratively reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

    Description from the original documentation:

    Baseline drift always blurs or even swamps signals and deteriorates analytical
    results, particularly in multivariate analysis.  It is necessary to correct
    baseline drift to perform further data analysis. Simple or modified polynomial
    fitting has been found to be effective in some extent. However, this method
    requires user intervention and prone to variability especially in low
    signal-to-noise ratio environments. The proposed adaptive iteratively
    reweighted Penalized Least Squares (airPLS) algorithm doesn't require any
    user intervention and prior information, such as detected peaks. It
    iteratively changes weights of sum squares errors (SSE) between the fitted
    baseline and original signals, and the weights of SSE are obtained adaptively
    using between previously fitted baseline and original signals. This baseline
    estimator is general, fast and flexible in fitting baseline.

    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        porder:
            integer indicating the order of the difference of penalties

    output:
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = whittaker_smooth(x, w, lam, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax):
                print('airpls: max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak,
        # so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    # print(i,'iterations used')
    return z
