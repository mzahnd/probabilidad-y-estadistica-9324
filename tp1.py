import math
import statistics
from tabulate import tabulate
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


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
    print("Variance\t", round(pgc_variance, 2))
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

    outliers_below = valuesBelow(pgc, tukey_limits[0])  # Outliers abajo
    outliers_above = valuesAbove(pgc, tukey_limits[1])  # Outliers arriba

    total_outliers = len(outliers_above) + len(outliers_below)
    percent_outliers = total_outliers / len(pgc)

    print("Porcentaje de outliers: " +
          str(round(percent_outliers * 100, 2)) + "%")

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


def poundToKilogram(pounds):
    """---"""

    return pounds*0.45359237


def inchesToCentimeters(inches):
    """"---"""

    return inches*2.54


def ejercicioE(df):
    """---"""

    printHeader('E')
    peso_lb = df["PESO"].tolist()
    altura_in = df["ALTURA"].tolist()

    # Convierto los datos de lb a kg y pulgadas a cm, según corresponda
    # Como cada medida es convertida y guardada en el índice, se mantiene
    # la relación entre los datos
    peso_kg = [poundToKilogram(p) for p in peso_lb]
    altura_cm = [inchesToCentimeters(a) for a in altura_in]

    plt.plot(altura_cm, peso_kg, 'bo')
    plt.xlabel("Altura [cm]")
    plt.ylabel("Peso [kg]")
    plt.show()


def ejercicioF(df):
    """"---"""

    printHeader('F')

    imc = df["IMC"].tolist()

    # Datos con IMC mayores a 25 y 30
    above25p = 0
    above30p = 0
    for i in imc:
        if i > 30:
            above30p += 1
            above25p += 1
        elif i > 25:
            above25p += 1

    above25p /= len(imc)
    above30p /= len(imc)

    print("Personas con un IMC mayor a 25: "
          + str(round(above25p*100, 2)) + "%")
    print("Personas con un IMC mayor a 30: "
          + str(round(above30p*100, 2)) + "%")

    plt.plot(list(range(1, len(imc) + 1)), imc, 'ro')
    plt.suptitle('IMC: Grafico valor vs caso')
    plt.show()


def ejercicioG(df):
    """---"""

    printHeader('G')
    pgc = df["PGC"].tolist()
    imc = df["IMC"].tolist()

    plt.plot(imc, pgc, 'ro')
    plt.xlabel("IMC")
    plt.ylabel("PGC")
    plt.show()


def ejercicioH_I(df):
    """---"""

    printHeader('H | I')
    pgc = df["PGC"]

    corr_coef = dict()

    for (nombreColumna, datosColumna) in df.iteritems():
        # Evitar columnas N y PGC
        if nombreColumna == "N" or nombreColumna == "PGC":
            continue

        # La biblioteca _pandas_ tiene una función llamada corr()
        # Utilizando el metodo de Pearson, es equivalente a corr() en Ocatave
        p = pgc.corr(datosColumna, method='pearson')

        # Guardo el valor de p
        corr_coef[nombreColumna] = p

        print("Correlación entre PGC y " + nombreColumna + ":  "
              + str(round(p, 2)))

    # Máximo p, despreciando la densidad
    p_maximo = ['', 0]
    for nombre, p in corr_coef.items():
        # Ignoro la densidad
        if (nombre.lower() == "densidad"):
            continue

        if (abs(p) > p_maximo[1]):
            p_maximo[0] = nombre
            p_maximo[1] = abs(p)

    print("Max: ")
    print(p_maximo)

    plt.plot(df[p_maximo[0]], pgc, 'ro')
    plt.xlabel(p_maximo[0])
    plt.ylabel("PGC")
    plt.show()


def ejercicioJ(df):
    """---"""
    
    printHeader('J')
    pgc = df["PGC"].tolist()
    abdomen = df["ABDOMEN"].tolist()

    # TODO: Conseguir datos reales
    mediciones_alumnos = [123, 423, 653, 954, 135]
    pgc_alumnos = np.polyval(np.polyfit(abdomen, pgc, 1), mediciones_alumnos)

    print(pgc_alumnos)


def ejercicioK(df):
    """---"""

    printHeader('K | L')

    pgc_all = df["PGC"].tolist()
    abdomen_all = df["ABDOMEN"].tolist()
    imc = df["IMC"].tolist()
    peso_lb = df["PESO"]

    peso_kg = [(n, poundToKilogram(p)) for n, p in peso_lb.iteritems()]

    peso_under_70 = list()
    peso_overeq_70 = list()

    peso_under_70_pgc = list()
    peso_under_70_abdomen = list()
    peso_overeq_70_pgc = list()
    peso_overeq_70_abdomen = list()

    for n, w in peso_kg:
        if w < 70:
            peso_under_70.append(n)
            peso_under_70_pgc.append(pgc_all[n])
            peso_under_70_abdomen.append(abdomen_all[n])
        else:
            peso_overeq_70.append(n)
            peso_overeq_70_pgc.append(pgc_all[n])
            peso_overeq_70_abdomen.append(abdomen_all[n])

    peso_under_70_imc25p = len(valuesAbove(peso_under_70, 25)) \
        / len(peso_under_70)
    peso_overeq_70_imc25p = len(valuesAbove(peso_overeq_70, 25)) \
        / len(peso_overeq_70)

    print("U 70: " + str(round(peso_under_70_imc25p*100, 2)))
    print("OE 70: " + str(round(peso_overeq_70_imc25p*100, 2)))

    plt.plot(peso_under_70_abdomen, peso_under_70_pgc,
             'g^', label='Peso menor a 70 kg')
    plt.plot(peso_overeq_70_abdomen, peso_overeq_70_pgc,
             'ro', label='Peso mayor o igual a 70 kg')

    plt.legend()
    plt.xlabel("Abdomen")
    plt.ylabel("PGC")
    plt.show()


def estimarPGC(h, a, n, sexo):
    """---
    sexo: 'M' (masculino) o 'F'(femenino)
    """
    est = 0
    if sexo == 'M':
        est = 10.1 - 0.239 * h + 0.8 * a - 0.5 * n
    elif sexo == 'F':
        est = 19.2 - 0.239 * h + 0.8 * a - 0.5 * n

    return est


def ejercicioM(df):
    """---"""

    printHeader('M')
    pgc = df["PGC"].tolist()
    altura_in = df["ALTURA"].tolist()
    abdomen = df["ABDOMEN"].tolist()
    cuello = df["CUELLO"].tolist()

    altura_cm = [inchesToCentimeters(a) for a in altura_in]

    # Desconozco el sexo de los individuos. Estimo el PGC con ambos sexos
    pgc_est_hombres = list()
    pgc_est_mujeres = list()
    for i in range(len(pgc)):
        pgc_est_hombres.append(estimarPGC(altura_cm[i],
                                          abdomen[i],
                                          cuello[i],
                                          'M'))

        pgc_est_mujeres.append(estimarPGC(altura_cm[i],
                                          abdomen[i],
                                          cuello[i],
                                          'F'))

    n_pgc = list(range(1, len(pgc) + 1))
    plt.plot(n_pgc, pgc,
             'ro', label='PGC Medido')
    plt.plot(n_pgc, pgc_est_hombres,
             'g^', label='PGC Estimado: población masculina')
    plt.plot(n_pgc, pgc_est_mujeres,
             'bs', label='PGC Estimado: población femenina')

    plt.legend()
    plt.ylabel("PGC")
    plt.show()


if __name__ == "__main__":
    datos = pd.read_csv('grasacorp.txt', delimiter='\t')
    # df.iloc[row_index,col_index]

    print()  # Linea en blanco
    ejercicioA(datos)
    print()  # Linea en blanco
    ejercicioB(datos)
    print()  # Linea en blanco
    ejercicioC(datos)
    print()  # Linea en blanco
    ejercicioD(datos)
    print()  # Linea en blanco
    ejercicioE(datos)
    print()  # Linea en blanco
    ejercicioF(datos)
    print()  # Linea en blanco
    ejercicioG(datos)
    print()  # Linea en blanco
    ejercicioH_I(datos)
    print()  # Linea en blanco
    ejercicioJ(datos)
    print()  # Linea en blanco
    ejercicioK(datos)
    print()  # Linea en blanco
    ejercicioM(datos)
    print()  # Linea en blanco
