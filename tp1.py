import math
import statistics
from tabulate import tabulate
from matplotlib import pyplot as plt
import pandas as pd


def printHeader(letra):
    print("Ejercicio " + letra)
    print("="*30)

def getArr(dataFrame, columnIndex):
    mean_arr = list()

    for i in range(len(dataFrame)):
        mean_arr.append(dataFrame.iloc[i, columnIndex])

    return mean_arr


def get_range(data):
    """..."""
    max_val = 0
    min_val = 0
    for element in data:
        if element > max_val:
            max_val = element
        elif element < min_val:
            min_val = element

    return max_val - min_val


def get_mode(data, retrieve='all'):
    """Calculate the mode and return None if all elements are equal.
    data: Array of data (will be passed to statistics.multimode).
    retrieve: Only applies when there's more than one mode.
        Can be any of the following:
                'all' (default): Array with all of them.
                'max': The maximum mode found.
                'min': The minimum mode found.
    """
    if retrieve not in ('all', 'max', 'min'):
        retrieve = 'all'
        raise RuntimeWarning("Invalid retrieve value!. "
                             + "Will retrieve all modes found.")

    tmp = statistics.multimode(data)
    if tmp == data:
        return None

    if len(tmp) > 1:
        if retrieve == 'max':
            tmp = max(tmp)
        elif retrieve == 'min':
            tmp = min(tmp)
    else:
        tmp = tmp[0]  # Retrieve a single number

    return tmp


def get_hist_bins(data, minimum=None, maximum=None, groups=None):
    """..."""
    if not minimum:
        minimum = math.floor(min(data))
    if not maximum:
        maximum = math.ceil(max(data))
    if not groups:
        groups = round(math.log2(len(data)))

    bins = list()
    for current in range(groups):
        bins.append(minimum + current*groups)

    bins.append(maximum)

    return bins


def get_points_frequency_polygon(bins_value, bins):
    """..."""
    n_bins = len(bins)  # Number of bins = length of array
    mean_half_dist_between_bins = 0
    points_x = [0.0]
    points_y = [0.0]  # bins_value  with coords 0.0 at the beggining and end
    for value in bins_value:
        points_y.append(value)
    points_y.append(0.0)

    # Start from the second element
    for index in range(n_bins-1):
        mean_half_dist_between_bins += (bins[index+1] - bins[index]) / 2
        points_x.append((bins[index+1] + bins[index]) / 2)

    # The first point is extrapolated based on the average of distance between
    # beans used in the previous points.
    mean_half_dist_between_bins /= n_bins-1
    points_x[0] = bins[0] - mean_half_dist_between_bins
    # Same idea por the last point
    points_x.append(bins[-1] + mean_half_dist_between_bins)

    print("X axis:", points_x)
    print("Y axis:", points_y)

    for point in points_x:
        point = float(point)
    for point in points_y:
        point = float(point)

    # verts = list(zip(points_x, points_y))
    # codes = []
    # for _ in verts:
    #     codes.append(Path.LINETO)
    # codes[0] = Path.MOVETO

    # print(verts)
    # print(codes)
    #
    # path = Path(verts, codes)
    #
    # patch = patches.PathPatch(path, facecolor='none', lw=1)
    # ax.add_patch(patch)
    # ax.set_xlim(0, 50)
    # ax.set_ylim(0, 30)
    # # plt.show()

    return points_x, points_y

def ejercicioA(df):
    """df: Datos del archivo"""
    printHeader('A')
    # Datos
    pgc_raw = df["PGC"].tolist()
    pgc_raw.sort()


    # Media
    pgc_mean = statistics.mean(pgc_raw)

    # Mediana
    pgc_median = statistics.median(pgc_raw)

    # Dispersion
    # Varianza
    pgc_variance = statistics.variance(pgc_raw)
    # Desviacion
    pgc_std = math.sqrt(pgc_variance)

    # Max,min
    pgc_max = max(pgc_raw)
    pgc_min = min(pgc_raw)

    # Quantiles
    pgc_quantiles = statistics.quantiles(pgc_raw, n=4)
    pgc_q1 = pgc_quantiles[0]   # q1
    pgc_q3 = pgc_quantiles[2]   # q3

    print("Mean\t\t", round(pgc_mean, 2))
    print("Median\t\t", round(pgc_median, 2))
    print("Variance\t",round(pgc_variance, 2))
    print("STD\t\t", round(pgc_std, 2))
    print("Max\t\t", round(pgc_max, 2))
    print("Min\t\t", round(pgc_min, 2))
    print("Q1\t\t", round(pgc_q1, 2))
    print("Q3\t\t", round(pgc_q3, 2))


def ejercicioB(df):
    """---"""

    printHeader('B')

    pgc = df["PGC"].tolist()
    n_pgc = list(range(1, len(df) + 1))
    
    ones = list()
    for i in range(len(df)):
        ones.append(1)

    plot1 = plt.figure(1)
    plt.plot(n_pgc, pgc, 'ro')
    plt.suptitle('PGC: Grafico valor vs caso')
    
    plot2 = plt.figure(2)
    plt.plot(pgc, ones, 'ro')
    plt.suptitle('PGC: Grafico constante vs valor')
    
    plt.show()


def valuesAbove(data, limit):
    res = list()
    for i in data:
        if i > limit:
            res.append(i)
    
    return res


def valuesBelow(data, limit):
    res = list()
    for i in data:
        if i < limit:
            res.append(i)
    
    return res

def ejercicioC(df):
    """---"""

    printHeader('C')

    pgc = df["PGC"].tolist()

    # Calculo outliers
    pgc_quantiles = statistics.quantiles(pgc, n=4)
    
    # iqr = q3 - q1
    iqr = pgc_quantiles[2] - pgc_quantiles[0]
    # q1 - 1.5 iqr  y  q3 + 1.5 iqr
    
    tukey_limits = [pgc_quantiles[0] - 1.5 * iqr, pgc_quantiles[0] + 1.5 * iqr]

    outliers_below = valuesBelow(pgc, tukey_limits[0]) # Outliers abajo
    outliers_above = valuesAbove(pgc, tukey_limits[1]) # Outliers arriba

    total_outliers = len(outliers_above) + len(outliers_below)
    percent_outliers = total_outliers / len(pgc)

    print("Porcentaje de outliers: " + str(round(percent_outliers * 100, 2)) + "%")


    plt.boxplot(pgc)
    plt.suptitle('PGC: Boxplot')
    plt.show()


def ejercicioD(df):
    printHeader('D')

    pgc = df["PGC"].tolist()
    pgc.sort()

    bins = get_hist_bins(pgc, minimum=15, maximum=40, groups=5)

    fig, ax = plt.subplots()
    n, bins2, patches = plt.hist(pgc, bins=bins, density=False, color="grey")

    freq_pol_x, freq_pol_y = get_points_frequency_polygon(n, bins)

    ax.plot(freq_pol_x, freq_pol_y, linestyle='--', lw=2, color="blue")
    plt.show()


if __name__ == "__main__":
    datos = pd.read_csv('grasacorp.txt', delimiter = '\t')
    #df.iloc[row_index,col_index]
    print() # Linea en blanco
    ejercicioA(datos)
    print() # Linea en blanco
    ejercicioB(datos)
    print() # Linea en blanco
    ejercicioC(datos)
    print() # Linea en blanco
    ejercicioD(datos)
    print() # Linea en blanco