#!/bin/python3

import math
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


def meanGreaterLower(alpha, data_list):
    x = data_list
    n = len(x)
    xbar = np.mean(x)
    sn = st.sem(x)
    if (alpha > 0.5):
        alpha = 1 - alpha

    t = st.t.ppf(1-alpha, n-1)

    return [ xbar - t*sn, xbar + t*sn ]


def meanGreater2(alpha, xbar, n, s2):
    #x = data_list
    sn = math.sqrt(s2/n)
    if (alpha > 0.5):
        alpha = 1 - alpha

    t = st.t.ppf(1-alpha, n-1)

    return xbar - t*sn


def getMeanGreater(alpha, data_list):
    return meanGreaterLower(alpha, data_list)[0]


def getMeanLower(alpha, data_list):
    return meanGreaterLower(alpha, data_list)[1]


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


def getPValue(desired_mean, data_list):
    x = data_list
    n = len(x)

    xbar = np.mean(x)
    S2 = 1/(n-1) * sum([i**2 for i in x]) - n/(n-1) * xbar*xbar

    t_obs = (xbar - desired_mean)/(math.sqrt(S2/n))
    # 1 - cdf
    p_val = st.t.sf(t_obs, n-1)

    return t_obs, p_val


def getP2Value(desired_mean, n, xbar, S2):
    t_obs = (xbar - desired_mean)/(math.sqrt(S2/n))
    # 1 - cdf
    p_val = st.t.sf(t_obs, n-1)

    return t_obs, 2*p_val


if __name__ == "__main__":
    # Media deseada (para calcular el valor p)
    desired_mean = 7.5
    # Confianza
    alpha = 0.95
    # Datos
    x = [5.49, 5.43, 5.35, 5.44, 5.36, 5.52, 5.64, 5.45, 5.75, 5.32]

    ans = getRangeMean(alpha, x)
    print("Mean:\t[{:0.5f}, {:0.5f}]".format(
        round(ans[0], 5), round(ans[1], 5)
        ))

    ans = getMeanGreater(alpha, x)
    print("Mean >:\t{:0.5f}".format( round(ans, 5) ))

    ans = getMeanLower(alpha, x)
    print("Mean <:\t{:0.5f}".format( round(ans, 5) ))
    print()

    ans = getRangeVar(alpha, x)
    print("Var:\t[{:0.5f}, {:0.5f}]".format(
        round(ans[0], 5), round(ans[1], 5)
        ))
    print()

    # p value
    t_obs, p_val = getPValue(desired_mean, x)
    print("t_obs:\t{:0.5f}".format(round(t_obs, 5)))
    print("p_val:\t{:0.5f}".format(round(p_val, 5)))
    print()

    desired_mean = 50
    n = 50
    xbar = 51.33
    S2 = 3.19**2

    t_obs, p_val = getP2Value(desired_mean, n, xbar, S2)
    print("t_obs:\t{:0.5f}".format(round(t_obs, 5)))
    print("p_val:\t{:0.5f}".format(round(p_val, 5)))

    p_val = meanGreater2(0.1, 1.4, 20, 0.4) 
    print("p_val:\t{:0.5f}".format(round(p_val, 5)))
