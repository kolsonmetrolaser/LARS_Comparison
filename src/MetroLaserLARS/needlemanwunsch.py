# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:58:36 2024

@author: KOlson

"""
# External imports
import numpy as np
from numpy.typing import NDArray
from numba import njit
from time import time


@njit
def needleman_wunsch(x: NDArray, y: NDArray, penalty_order: float = 1, gap: float = 1, insert: float = -1,
                     nw_normalized: bool = False):
    """
    Uses an adapted Needleman-Wunsch algorithm to calculate matches between two arrays of numbers.

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
    insert : float, optional
        Value to be inserted into x and y where non-matches occur. The default is -1.
    normalized: bool, optional
        Whether the distance used to calculate the penalty should be normalized. The default is false.
    Returns
    -------
    NDArray
        Array made up of `x`, with `insert` inserted where there is no match with `y`.
    NDArray
        Array made up of `y`, with `insert` inserted where there is no match with `x`.
    float
        The quality of the fit, equal to the sum of penalties for matches and non-matches.

    """
    nx, ny = len(x), len(y)
    # Optimal score
    F = np.zeros((nx+1, ny+1))
    F[:, 0] = np.linspace(0, -nx*gap**penalty_order, nx+1)
    F[0, :] = np.linspace(0, -ny*gap**penalty_order, ny+1)
    # Pointers
    P = np.zeros((nx+1, ny+1))
    P[:, 0] = 3
    P[0, :] = 4

    # Temp scores
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            t[0] = F[i, j] - (2*abs(x[i]-y[j])/(x[i]+y[j]))**penalty_order\
                if nw_normalized else F[i, j] - abs(x[i]-y[j])**penalty_order
            t[1] = F[i, j+1] - gap**penalty_order
            t[2] = F[i+1, j] - gap**penalty_order
            tmax = np.max(t)
            F[i+1, j+1] = tmax
            P[i+1, j+1] += 2 if t[0] == tmax else 0
            P[i+1, j+1] += 3 if t[1] == tmax else 0
            P[i+1, j+1] += 4 if t[2] == tmax else 0

    # Trace through optimal alignment
    i = nx
    j = ny
    rx = insert*np.ones(nx+ny)
    ry = insert*np.ones(nx+ny)
    ri = 0
    while i > 0 or j > 0:
        if P[i, j] in [2, 5, 6, 9]:
            rx[ri] = x[i-1]
            ry[ri] = y[j-1]
            i -= 1
            j -= 1
        elif P[i, j] in [3, 7]:  # original code has [3,5,7,9] but 5 and 9 are checked above
            rx[ri] = x[i-1]
            i -= 1
        elif P[i, j] == 4:  # original code has [4,6,7,9] but 6,7,9 are checked above
            ry[ri] = y[j-1]
            j -= 1
        ri += 1
    rx, ry = rx[:ri], ry[:ri]
    result = rx[::-1], ry[::-1], F[-1, -1]
    return result

# About the same with or without njit (better for stretching_iterations~>30)


@njit
def find_matches(x: NDArray, y: NDArray, max_stretch: float = 0.02, num_stretches: int = 1000,
                 stretching_iterations: int = 5, stretch_iteration_factor: float = 5,
                 penalty_order: float = 1, gap: float = 1, insert: float = -1, nw_normalized: bool = False):
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
    bestrx, bestry, bestq = needleman_wunsch(x, y, penalty_order=penalty_order, gap=gap, insert=insert,
                                             nw_normalized=nw_normalized)
    best_stretch = 1
    search_space_delta = 0
    for stretching_iteration in range(stretching_iterations):
        if stretching_iteration == 0:
            search_space = np.linspace(1-max_stretch, 1+max_stretch, num_stretches)
            search_space_delta = 2*max_stretch/(num_stretches-1)
        else:
            search_space = np.linspace(best_stretch-(num_stretches/stretch_iteration_factor/2)*search_space_delta,
                                       best_stretch+(num_stretches/stretch_iteration_factor/2)*search_space_delta,
                                       num_stretches+2)
            search_space_delta = 2*(num_stretches/stretch_iteration_factor/2)*search_space_delta/(num_stretches+1)
            search_space = search_space[1:-1]
        # This is actually better than vectorizing the problem and doing many matches at once (numba magic)
        for i, s in enumerate(search_space):
            rx, ry, q = needleman_wunsch(x, s*y, penalty_order=penalty_order, gap=gap, insert=insert,
                                         nw_normalized=nw_normalized)
            bestq = q if q > bestq else bestq
            best_stretch = s if q == bestq else best_stretch
            bestrx, bestry = (rx, ry) if q == bestq else (bestrx, bestry)
    return bestrx, bestry, bestq, best_stretch, search_space_delta


if __name__ == '__main__':
    # TESTING

    from time import time
    max_stretch = .02  # Maximum allowed stretching factor. The second folder is allowed to stretch at most by a factor of (1±`max_stretching_factor`)
    num_stretches = 1000  # Number of different stretching factors to check each iteration
    stretching_iterations = 5  # How many iterations deep to check stretching factors.
    stretch_iteration_factor = 5
    max_mismatch = 150*2/(60000+10000)  # .005
    penalty_order = 1

    time0 = time()
    x = np.array([13721.5, 14171, 15149, 16384, 16655, 16950.5, 17458, 18085, 18834, 20258, 20596, 22462.5, 23114.5, 23899.5, 24966, 25426, 27982, 28221.5, 28348.5, 28652, 29847.5, 31144, 31509, 31629, 32453.5, 32613.5, 33936, 34093.5, 34512.5, 34849.5, 35718.5, 36055, 36272, 36558, 37313.5, 37724, 38125, 39618.5, 40320, 43337.5, 46304.5, 46704, 47498.5, 48644, 48807, 49003.5, 49450.5, 49702, 50224, 50642, 52103.5, 53225.5, 53802, 54307, 54601, 55286, 55503.5, 55812, 56052, 56329, 57515, 57833.5, 58248.5, 58741.5, 59014, 59511])
    y = np.array([13733.5, 14158, 15149, 16741.5, 18055, 18850, 20284, 20602.5, 22480, 23094.5, 23858.5, 24981, 25435, 27989, 28227, 28669.5, 29842, 31140.5, 31530, 31633.5, 32463.5, 33925, 34114.5, 34512.5, 34903, 35733, 35864, 36096, 36284.5, 36549, 37307.5, 37751.5, 38161.5, 39646.5, 40366, 43139, 43377, 46313.5, 46733.5, 47523, 48677, 48820, 49026, 49471.5, 49713.5, 50225.5, 50674.5, 52117, 53249.5, 53837, 54318.5, 54622.5, 55507, 56098, 56354.5, 57577.5, 57834.5, 58269.5, 58780, 59064, 59580.5])
    # x = np.array([13700, 23800, 33900, 44000, 54100, 64200])
    # y = np.array([13749, 23850, 33951, 44099, 54200, 64301])

    time0 = time()
    kwargs_find_matches = {'max_stretch': max_stretch, 'num_stretches': num_stretches,
                           'stretching_iterations': stretching_iterations,
                           'stretch_iteration_factor': stretch_iteration_factor, 'penalty_order': penalty_order,
                           'gap': max_mismatch/2, 'nw_normalized': True}
    bestrx, bestry, bestq, best_stretch, best_stretch_error = find_matches(x, y, **kwargs_find_matches)
    print(f'Done after {time()-time0} s')

    # print(f'Best Quality: {bestq:.2f} at stretch {best_stretch:.6f} p/m {best_stretch_error}')
    # print(f'unmatched: {np.sum(bestrx == -1)+np.sum(bestry == -1)}')
    # for x, y in zip(bestrx, bestry):
    #     print(f'{x:7.1f}  {best_stretch*y:7.1f}')
