"""Guía 0."""

import math
import statistics
from tabulate import tabulate
from matplotlib import pyplot as plt
# from itertools import islice


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
        groups = round(math.log2(data.len()))

    bins = list()
    for current in range(groups):
        bins.append(minimum + current*groups)

    bins.append(maximum)

    return bins


def ejercicio1():
    """..."""
    data = [1067, 919, 1196, 785, 1126, 936, 918, 1156, 920, 948]

    data.sort()
    print("Datos ordenados:", data)

    print("Media:", statistics.mean(data))
    print("Mediana:", statistics.median(data))
    print("Rango:", get_range(data))
    print("Desviación estándar:", statistics.stdev(data))
    quarts = statistics.quantiles(data, n=4)
    print("Cuartiles:", quarts[0], quarts[1])


def ejercicio2():
    """..."""
    data = [
        9, 8, 3, 18, 4, 5, 6, 7, 7, 6, 7, 5, 4, 3, 15,
        3, 8, 6, 11, 10, 9, 8, 7, 13, 3, 4, 5, 5, 6, 4,
        3, 6, 7, 9, 8, 7, 4, 5, 6, 7, 8, 10, 11, 3, 2,
        1, 7, 6, 17, 7, 9, 8, 6, 11, 0, 20, 1, 4, 5, 12,
        2, 2, 1, 4, 5, 6, 7, 8, 10, 9, 8, 7, 7, 6, 5,
        2, 7, 7, 10, 6, 6, 14, 2, 4, 5, 12, 10, 9, 8, 7
    ]

    # It is important to sort the data and calculate its sum before doing
    # anything else because the algorithm takes advantage of this
    # (aka it needs it)
    data.sort()
    data_sum = sum(data)
    data_mean = statistics.mean(data)  # Will be used later

    # Number of replacements is the key.
    # Value is an array [sum, relative frequency]
    weekly_replacements = dict()
    # Positions in value array
    WSUM = 0
    WFREQ = 1

    # Create the first key-value pair.
    # Note that the value array is created as empty
    previous_value = data[0]
    weekly_replacements[data[0]] = [0, -1]
    for value in data:
        # Calculate the frecuency of the previously calculated week
        # This is valid because data has been SORTED previously.
        if value != previous_value:
            weekly_replacements[previous_value][WFREQ] = \
                weekly_replacements[previous_value][WSUM]/data_sum
            # New founded
            weekly_replacements[value] = [0, -1]
            previous_value = value

    # Add to ammount of weeks with this number (value) of replacements
        weekly_replacements[value][WSUM] += 1

    # After exiting the loop, the relative frequency of the last element
    # remains uncalculated.
    weekly_replacements[previous_value][WFREQ] = \
        weekly_replacements[previous_value][WSUM]/data_sum

    print(tabulate([[k, ] + v for k, v in weekly_replacements.items()],
                   headers=["Repuestos semanales",
                            "Cantidad de semanas",
                            "Frecuencia relativa"]
                   )
          )

    print("Media:", data_mean)
    print("Mediana:", statistics.median(data))
    print("Varianza:", statistics.variance(data, xbar=data_mean))
    print("Desviación estándar:", statistics.stdev(data, xbar=data_mean))
    print("Moda:", get_mode(data))


def ejercicio3():
    """..."""
    data = [
        32.5, 15.2, 35.4, 21.3, 28.4, 26.9, 34.6, 29.3, 24.5, 31.0,
        21.2, 28.3, 27.1, 25.0, 32.7, 29.5, 30.2, 23.9, 23.0, 26.4,
        27.3, 33.7, 29.4, 21.9, 29.3, 17.3, 29.0, 36.8, 29.2, 23.5,
        20.6, 29.5, 21.8, 37.5, 33.5, 29.6, 26.8, 28.7, 34.8, 18.6,
        25.4, 34.1, 27.5, 29.6, 22.2, 22.7, 31.3, 33.2, 37.0, 28.3,
        36.9, 24.6, 28.9, 24.8, 28.1, 25.4, 34.5, 23.6, 38.4, 24.0
    ]
    data.sort()
    bins = get_hist_bins(data, min=15, max=40, groups=5)

    fig, ax = plt.subplots()
    n, bins2, patches = plt.hist(data, bins=bins, density=False, color="grey")

    print(n)
    print(bins2)
    print(patches)

    freq_pol_x, freq_pol_y = get_points_frequency_polygon(n, bins)

    ax.plot(freq_pol_x, freq_pol_y, linestyle='--', lw=1, color="blue")
    plt.xlabel('Resistencia a la ruptura [oz]')
    plt.show()


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


if __name__ == '__main__':
    ejercicio3()
