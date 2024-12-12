# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:58:36 2024

@author: KOlson

"""
# External imports
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from time import time


@njit
def needleman_wunsch(x: NDArray, y: NDArray, penalty_order: float = 1, gap: float = 1, nw_normalized: bool = False):
    """
    Uses an adapted Needleman-Wunsch algorithm to calculate matches between two arrays of numbers.
    https://en.wikipedia.org/wiki/Needleman-Wunsch_algorithm

    Code adapted from "A simple version of the Needleman-Wunsch algorithm in Python."
    See https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5

    Parameters
    ----------
    x : NDArray
        First array to be matched.
    y : NDArray
        Second array to be matched.
    penalty_order : float, optional
        Order of distance penalty. If 1, the penalty is the difference between two numbers,
        if 2, the penalty is the square difference, etc. The default is 1.
    gap : float, optional
        The penalty for not matching is `gap**penalty_order`. `gap` corresponds the maximum allowed difference
        between matched values. The default is 1.
    normalized: bool, optional
        Whether the distance used to calculate the penalty should be normalized. The default is false.
    Returns
    -------
    NDArray
        Array made up of `x`, with -1 inserted where there is no match with `y`.
    NDArray
        Array made up of `y`, with -1 inserted where there is no match with `x`.
    float
        The quality of the fit, equal to the sum of penalties for matches and non-matches.

    """
    nx, ny = len(x), len(y)
    # Store the optimal scores
    F = np.zeros((nx+1, ny+1))
    F[:, 0] = np.linspace(0, -nx*gap**penalty_order, nx+1)
    F[0, :] = np.linspace(0, -ny*gap**penalty_order, ny+1)
    # Pointers keep track of path through optimal scores
    P = np.zeros((nx+1, ny+1))
    P[:, 0] = 3
    P[0, :] = 4

    # Temporary scores
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            # next score is previous score plus the distance penalty (t[0]) or a non-match penalty (t[1] or t[2])
            t[0] = F[i, j] - (2*abs(x[i]-y[j])/(x[i]+y[j]))**penalty_order\
                if nw_normalized else F[i, j] - abs(x[i]-y[j])**penalty_order
            t[1] = F[i, j+1] - gap**penalty_order
            t[2] = F[i+1, j] - gap**penalty_order
            tmax = np.max(t)
            F[i+1, j+1] = tmax
            # keep track of path to take through the optimal scores
            # sums of permutations of 2, 3, and 4 are unique, so this is an efficient way to store this info
            # when multiple options are identical, arbitrary choices are made later (favoring matching)
            P[i+1, j+1] += 2 if t[0] == tmax else 0
            P[i+1, j+1] += 3 if t[1] == tmax else 0
            P[i+1, j+1] += 4 if t[2] == tmax else 0

    # Retrace the path through optimal alignment, starting from the bottom right corner
    i = nx
    j = ny
    rx = -np.ones(nx+ny)
    ry = -np.ones(nx+ny)
    ri = 0
    while i > 0 or j > 0:  # until reaching the top left, keep traversing
        if P[i, j] in [2, 5, 6, 9]:  # if the peaks are matched, store the peaks and move up and left
            rx[ri] = x[i-1]
            ry[ri] = y[j-1]
            i -= 1
            j -= 1
        elif P[i, j] in [3, 7]:  # add an unmatched peak to x and move up
            rx[ri] = x[i-1]
            i -= 1
        elif P[i, j] == 4:  # add an unmatched peak to y and move left
            ry[ri] = y[j-1]
            j -= 1
        ri += 1
    rx, ry = rx[:ri], ry[:ri]  # cut off the lists to be the right length
    # we traversed the list backwards, so flip them. The overall quality is in the bottom-right of F
    result = rx[::-1], ry[::-1], F[-1, -1]
    return result


@njit(parallel=True)
def find_matches_inner_parallel(x, y, ssmin, ssmax, ssdelta, penalty_order, gap, nw_normalized):
    # comments: see find_matches_inner_serial
    n = int((ssmax-ssmin)/ssdelta)
    qs = np.zeros(n)
    for i in prange(n):
        _, _, qs[i] = needleman_wunsch(x, (ssmin+ssdelta*i)*y,
                                       penalty_order=penalty_order, gap=gap, nw_normalized=nw_normalized)
    return qs


def find_matches_inner_serial(x, y, ssmin, ssmax, ssdelta, penalty_order, gap, nw_normalized):
    n = int((ssmax-ssmin)/ssdelta)
    qs = np.zeros(n)
    # for each stretch...
    for i in range(n):
        # use needleman-wunsch algorithm to match x to stretched y, saving the quality to an array
        _, _, qs[i] = needleman_wunsch(x, (ssmin+ssdelta*i)*y,
                                       penalty_order=penalty_order, gap=gap, nw_normalized=nw_normalized)
    return qs


def find_matches(x: NDArray, y: NDArray, max_stretch: float = 0.02, num_stretches: int = 1000,
                 penalty_order: float = 1, gap: float = 1, nw_normalized: bool = False, **kwargs):
    """
    Finds matches between reference peaks and measured peaks, allowing stretching of the measured peaks.
    Tests many stretches, determines the best overall matching quality, and reports the matches and
    stretch of the best fit.

    Parameters
    ----------
    x : NDArray
        Reference peak locations.
    y : NDArray
        Measured peak locations.
    max_stretch : float, optional
        `y` is allowed to stretch from `1-max_stretch` to `1+max_stretch`. The default is 0.02.
    num_stretches : int, optional
        Number of stretches to test per iteration. The default is 1000.
    stretching_iterations : int, optional
        Number of stretch testing iterations. The default is 5.
    stretch_iteration_factor : float, optional
        Factor by which the stretching search space is reduced each iteration. The default is 5.
    kwargs:
        See `needleman_wunsch()` for other kwargs.

    Note: The final precision in the stretching factor is approximately
    `2*max_stretch/(stretch_iteration_factor**(stretching_iterations-1)*num_stretches)`

    Returns
    -------
    bestrx : NDArray
        Array made up of `x`, with `insert` inserted where there is no match with
        `y` corresponding to the best quality match.
    bestry : NDArray
        Array made up of `y`, with `insert` inserted where there is no match with
        `x` corresponding to the best quality match.
    bestq : float
        The quality of the best match, equal to the sum of penalties for matches and non-matches.
    best_stretch : float
        The stretch corresonding to the best quality match.
    search_space_delta : float
        The difference between `best_stretch` and the next closest stretch values tested.

    """
    bestq = -100000000
    best_stretch = 1
    search_space_delta = 0
    search_space = np.linspace(1-max_stretch, 1+max_stretch, num_stretches)
    search_space_delta = 2*max_stretch/(num_stretches-1)

    # parallel better for ~10_000 stretches or more
    if num_stretches > 10_000:
        qs = find_matches_inner_parallel(x, y, search_space[0], search_space[-1], search_space_delta,
                                         penalty_order, gap, nw_normalized)
    else:
        qs = find_matches_inner_serial(x, y, search_space[0], search_space[-1], search_space_delta,
                                       penalty_order, gap, nw_normalized)

    # find the stretch with the best quality
    best_stretch = search_space[np.where(qs == np.max(qs))[0][0]]

    # full calculation for the best stretch
    bestrx, bestry, bestq = needleman_wunsch(x, best_stretch*y, penalty_order=penalty_order, gap=gap, nw_normalized=nw_normalized)

    return bestrx, bestry, bestq, best_stretch, search_space_delta


if __name__ == '__main__':
    # TESTING

    max_stretch = .1  # Maximum allowed stretching factor. `y` is allowed to stretch at most by a factor of (1±`max_stretching_factor`)
    num_stretches = 10000  # Number of different stretching factors to check each iteration
    max_mismatch = .01  # 150*2/(60000+10000)  # .005
    penalty_order = 1

    # input data: numpy arrays of peak positions
    # x = np.array([13721.5, 14171, 15149, 16384, 16655, 16950.5, 17458, 18085, 18834, 20258, 20596, 22462.5, 23114.5, 23899.5, 24966, 25426, 27982, 28221.5, 28348.5, 28652, 29847.5, 31144, 31509, 31629, 32453.5, 32613.5, 33936, 34093.5, 34512.5, 34849.5, 35718.5, 36055, 36272, 36558, 37313.5, 37724, 38125, 39618.5, 40320, 43337.5, 46304.5, 46704, 47498.5, 48644, 48807, 49003.5, 49450.5, 49702, 50224, 50642, 52103.5, 53225.5, 53802, 54307, 54601, 55286, 55503.5, 55812, 56052, 56329, 57515, 57833.5, 58248.5, 58741.5, 59014, 59511])
    # y = np.array([13733.5, 14158, 15149, 16741.5, 18055, 18850, 20284, 20602.5, 22480, 23094.5, 23858.5, 24981, 25435, 27989, 28227, 28669.5, 29842, 31140.5, 31530, 31633.5, 32463.5, 33925, 34114.5, 34512.5, 34903, 35733, 35864, 36096, 36284.5, 36549, 37307.5, 37751.5, 38161.5, 39646.5, 40366, 43139, 43377, 46313.5, 46733.5, 47523, 48677, 48820, 49026, 49471.5, 49713.5, 50225.5, 50674.5, 52117, 53249.5, 53837, 54318.5, 54622.5, 55507, 56098, 56354.5, 57577.5, 57834.5, 58269.5, 58780, 59064, 59580.5])
    x = np.array([10966, 11173, 11703, 11987, 14038, 14202, 14497, 15237, 15513, 16699, 17027, 17228, 17804, 17859, 18571, 18887, 19480, 20064, 20848, 21086, 21111, 21637, 22466, 23049, 23888, 24455, 25161, 25614, 25674, 26107, 26256, 27925, 28638, 28998, 29018, 29436, 30255, 30810, 31692, 31982, 32287, 33238, 33458, 34045, 34891, 35198, 35589, 35606, 35998, 36169, 36716, 37186, 37220, 37446, 38024, 38380, 38814, 39246, 40544, 41084, 41460, 41795, 42500, 42571, 43238, 43519, 43600, 44034, 44205, 44531, 44955, 45664, 46164, 47025, 47312, 47417, 47598, 47935, 48604, 49386, 49625, 50151, 50395, 50803, 51275, 51721, 51886, 52363, 53118, 53325, 53643, 53871, 55117, 55282, 55713, 55743, 55955, 55996, 56372, 56512, 56628, 57115, 57274, 57591, 58398, 58848, 59163, 59411, 60063, 60473, 60972, 61231])
    y = np.array([10162.5, 10830.6, 11288.1, 13736.9, 14181, 15150.7, 16373.1, 16945.1, 17387.6, 17438.4, 18078.2, 18856.6, 20267.9, 20674.2, 22459.7, 23100.2, 23900.3, 24962.7, 25474.3, 27976.6, 28287.1, 28651.7, 29848.1, 30309.4, 31142.7, 31536.7, 32450.8, 33948.3, 34096.7, 34533, 34856.2, 35718.3, 36052.5, 36557.6, 37332.8, 37725.3, 38121.4, 39628.2, 40327.3, 43356.4, 46309.7, 46708.1, 47500.4, 48822.3, 49022.8, 49715.5, 50222.3, 50635.7, 52110.1, 53819.1, 54309.1, 54611.6, 55285.9, 56055.1, 56338.5, 57511.2, 58245.5, 58739.1, 59516])

    x = np.array([9611, 9992.7, 10966, 11173, 11703, 11987, 14038, 14202, 14497, 15237, 15513, 16699, 17027, 17228, 17804, 17859, 18571, 18887, 19480, 20064, 20848, 21086, 21111, 21637, 22466, 23049, 23888, 24455, 25161, 25614, 25674, 26107, 26256, 27925, 28638, 28998, 29018, 29436, 30255, 30810, 31692, 31982, 32287, 33238, 33458, 34045, 34891, 35198, 35589, 35606, 35998, 36169, 36716, 37186, 37220, 37446, 38024, 38380, 38814, 39246, 40544, 41084, 41460, 41795, 42500, 42571, 43238, 43519, 43600, 44034, 44205, 44531, 44955, 45664, 46164, 47025, 47312, 47417, 47598, 47935, 48604, 49386, 49625, 50151, 50395, 50803, 51275, 51721, 51886, 52363, 53118, 53325, 53643, 53871, 55117, 55282, 55713, 55743, 55955, 55996, 56372, 56512, 56628, 57115, 57274, 57591, 58398, 58848, 59163, 59411, 60063, 60473, 60972, 61231, 61277, 61445, 61617, 61832, 62443, 62919])
    y = 1000*np.array([10.1225, 10.6226, 11.3532, 11.5937, 13.7595, 14.2018, 15.1205, 16.2452, 16.6485, 17.3676, 17.9475, 18.2789, 18.817, 19.3429, 20.2207, 20.5766, 21.915, 22.2461, 23.0281, 23.8139, 24.9772, 25.3482, 27.914, 28.1736, 28.6301, 31.0675, 32.5033, 32.83, 33.8306, 34.4031, 34.8816, 36.0106, 37.6814, 38.1043, 40.3009, 40.6434, 41.4689, 41.9722, 42.5546, 43.0548, 43.2979, 46.221, 46.6345, 47.3825, 50.5755, 51.027, 54.5215, 54.7545, 55.2268, 56.2447, 58.1392, 57.468, 58.6702, 59.449])

    time0 = time()
    kwargs_find_matches = {'max_stretch': max_stretch, 'num_stretches': num_stretches,
                           'penalty_order': penalty_order,
                           'gap': max_mismatch/2, 'nw_normalized': True}
    bestrx, bestry, bestq, best_stretch, best_stretch_error = find_matches(x, y, **kwargs_find_matches)
    print(f'Done after {time()-time0} s')

    print(f'Best Quality: {bestq:.2f} at stretch {best_stretch:.6f} p/m {best_stretch_error}')
    print(f'x: {len(x)}, y:{len(y)}')
    print(f'matched: {len(x)+len(y)-len(bestrx)}, unmatched: {np.sum(bestrx == -1)+np.sum(bestry == -1)}')
    for x, y in zip(bestrx, bestry):
        print(f'{x:7.1f}  {y:7.1f}')
    print('all unique:')
    print(f'{[round(((x+y)/2 if x != -1 and y != -1 else (x if y == -1 else y)), 4) for x, y in zip(bestrx, bestry)]}')
