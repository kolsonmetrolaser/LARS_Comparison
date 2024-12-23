# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:51:32 2024

@author: KOlson
"""
import scipy.stats as stats
import numpy as np


def lepage_test(x, y):
    """
    Performs the Lepage test for location and scale differences between two samples.

    Args:
        x (array-like): First sample.
        y (array-like): Second sample.

    Returns:
        tuple: Lepage test statistic and p-value.
    """

    # Wilcoxon rank-sum test for location
    w_stat, _ = stats.ranksums(x, y)
    n1 = len(x)
    n2 = len(y)
    w_stat_std = (w_stat - n1 * (n1 + n2 + 1) / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    # Ansari-Bradley test for scale
    ab_stat, _ = stats.ansari(x, y)
    ab_stat_std = (ab_stat - n1 * (n1 + n2 + 1) / 4) / np.sqrt(n1 * n2 * (n1 + n2 + 1) * (n1 + n2 - 1) / (48 * (n1 + n2 - 2)))

    # Lepage test statistic
    lepage_stat = w_stat_std**2 + ab_stat_std**2

    # Approximate p-value using chi-square distribution
    p_value = 1 - stats.chi2.cdf(lepage_stat, 2)

    return lepage_stat, p_value


# Example usage
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]

lepage_stat, p_value = lepage_test(x, y)
print("Lepage statistic:", lepage_stat)
print("p-value:", p_value)
