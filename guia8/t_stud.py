#!/bin/python3

import numpy as np
from scipy import stats as st

def getRangeMean(alpha, data_list):
    x = data_list
    n = len(x)
    xbar = np.mean(x)
    sn = st.sem(x)
    if (alpha > 0.5):
        alpha = 1 - alpha

    t = st.t.ppf(1-alpha/2, n-1)

    return [xbar - t*sn, xbar + t*sn]

def getRangeVar(alpha, data_list):
    x = data_list
    n = len(x)

    if (alpha > 0.5):
        alpha = 1 - alpha

    xbar = np.mean(x)
    S2 = 1/(n-1) * sum([i**2 for i in x]) - n/(n-1) * xbar*xbar

    num = (n-1)*S2

    denom_left = st.chi2.ppf(1-alpha/2, n-1)
    denom_right = st.chi2.ppf(alpha/2, n-1)
    return [num/denom_left, num/denom_right]


if __name__ == "__main__":
    # Confianza
    alpha = 0.95
    
    # Datos
    x = [0.21, 0.19, 0.14, 0.19, 0.13, 0.16]

    ans = getRangeMean(alpha, x)
    print("Mean:\t", ans)

    ans = getRangeVar(alpha, x)
    print("Var:\t", ans)
