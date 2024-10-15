# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:49:38 2024

@author: KOlson
"""
import numpy as np
from numpy.typing import ArrayLike
from numpy.testing import assert_equal


def are_equal(first, second):
    try:
        assert_equal(first, second)
        return True
    except AssertionError:
        return False


def group(a: ArrayLike, maxsize: int = 1) -> list[ArrayLike]:
    """
    Groups `a` into subsets with size of at most `maxsize`.
    All but the last group are guaranteed to be length `maxsize`.

    Parameters
    ----------
    a : ArrayLike
        Array to split into smaller groups.
    maxsize : int
        Maximum size of the groups. The default is 1.

    Returns
    -------
    list[ArrayLike]
        List of grouped arrays.

    """
    a = a.copy()
    return_list = []
    while np.size(a) > 0:
        first_el = a[0]
        next_array = np.array(a[0])
        a = a[1:]
        while np.size(a) > 0 and a[0] < first_el+maxsize:
            next_array = np.append(next_array, a[0])
            a = a[1:]
        return_list.append(next_array)
    return return_list
