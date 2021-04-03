import math
import statistics
from tabulate import tabulate
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# Columnas del archivo con datos
COLUMNAS = [
    'N', 'PGC', 'Densidad', 'Edad', 'Peso', 'Altura',
    'IMC', 'Cuello', 'Pecho', 'Abdomen', 'Cadera',
    'Muslo', 'Rodilla', 'Tobillo', 'Biceps', 'Antebrazo',
    'Muñeca'
]


def printHeader(letra):
    print("Ejercicio " + letra)
    print("="*30)
    print()


def valuesAbove(data, limit):
    """Obtiene una lista con los valores mayores al límite.

    Args:
        data (list): Lista con todos los datos.
        limit (same as list data type): Valor sobre el cual se buscan
                                        los datos mayores.

    Returns:
        list: Valores en _data_ mayores a _limit_
    """

    res = list()
    for i in data:
        if i > limit:
            res.append(i)

    return res


def valuesBelow(data, limit):
    """Obtiene una lista con los valores menores al límite.

    Args:
        data (list): Lista con todos los datos.
        limit (same as list data type): Valor sobre el cual se buscan
                                        los datos menores.

    Returns:
        list: Valores en _data_ menores a _limit_
    """

    res = list()
    for i in data:
        if i < limit:
            res.append(i)

    return res


def libraKilogramo(pounds):
    """---"""

    return pounds*0.45359237


def pulgadasCentimetro(inches):
    """"---"""

    return inches*2.54


def puntos_poligono_frecuencia(bins_val, bins):
    """[summary]

    Esta función supone que los intervalos de clase son de igual amplitud.

    Args:
        bins_val (list): Altura de los rectángulos del historigrama,
                         devueltos por la función hist() de matplotlib.
        bins (list): En español, rectángulos, son los intervalos de clase
                     devueltos por la función hist() de matplotlib.

    Returns:
        float list, float list:
                    puntos del eje de abscisas, puntos del eje de ordenadas
    """

    # Cantidad de rectángulos (número de clases)
    n_bins = len(bins)
    # Ancho de los intervalos de clase.
    ancho_bins = (bins[0] + bins[1]) / 2

    # Puntos en X e Y.
    puntos_x = [0.0]            # Se modificará el 0.0 por la extrapolación
    puntos_y = [0.0]            # Un 0.0 como primer ordenada

    # Todas las alturas de los rectángulos son coordenadas en el
    # eje de ordenadas, agregando como coordenada extra el 0 al comienzo
    # y final de la lista (para poder crear luego el polígono de frecuencias).
    for value in bins_val:
        puntos_y.append(value)
    puntos_y.append(0.0)        # Un 0.0 al final de la lista

    # Se almacena la coordenada en X del punto medio de cada rectángulo.
    for index in range(n_bins-1):
        puntos_x.append((bins[index+1] + bins[index]) / 2)

    # El primer y último punto (con ordenada 0) son extrapolados en X tomando
    # como referencia la distancia media entre los rectángulos.
    puntos_x[0] = bins[0] - ancho_bins
    puntos_x.append(bins[-1] + ancho_bins)

    # Se convierten los datos a float para que matplotlib pueda utilizarlos.
    for point in puntos_x:
        point = float(point)
    for point in puntos_y:
        point = float(point)

    return puntos_x, puntos_y


def ejercicioA(df):
    """Item A

    Obtenga la media, mediana, dispersión, máximo, mínimo, primer y tercer
    cuartil de PGC.
    Muestre una tabla de estos parámetros redondeados en dos decimales.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """
    printHeader('A')

    # Datos
    pgc_raw = df["PGC"].tolist()

    # Ordenamos los datos
    pgc_raw.sort()

    # Los parámetros calculados se guardarán en este diccionario.
    pgc_parametros = dict()

    # Media
    pgc_parametros['Media muestral'] = statistics.mean(pgc_raw)

    # Mediana y Cuartiles
    # statistics.quantiles divide los datos en _n_ intervalos iguales,
    # devolviendo una lista de n-1 elementos.
    # Si _n_ = 4, se obtienen los cuartiles; _n_ = 10 los deciles; etc.
    # Notemos que con _n_ = 4, además de los cuartiles se obtiene la mediana.
    # Es decir, la función devuelve una lista con el formato:
    # [q1, mediana, q3]
    # Vamos a aprovechar esto para evitar repetir cálculos.
    # Otra opción para obtener la mediana seria utilizar la función
    # statistics.median().
    pgc_quantiles = statistics.quantiles(pgc_raw, n=4)

    pgc_parametros['Mediana'] = pgc_quantiles[1]          # Mediana
    pgc_parametros['Primer cuartil'] = pgc_quantiles[0]   # q1
    pgc_parametros['Tercer cuartil'] = pgc_quantiles[2]   # q3

    # Max,min
    pgc_parametros['Máximo'] = max(pgc_raw)
    pgc_parametros['Mínimo'] = min(pgc_raw)

    # Dispersion
    # Varianza
    # statistics.variance toma como argumento opcional (xbar) el valor de la
    # media. En caso de no proveer dicho valor, se calcula automáticamente.
    pgc_parametros['Varianza'] = statistics.variance(pgc_raw,
                                                     xbar=pgc_parametros[
                                                         'Media muestral'
                                                     ])
    # Desviacion
    # Sabemos que la desviación estándar es la raíz cuadrada de la varianza
    # muestral.
    # Podríamos utilizar la función statistics.stdev() para obtenerla, mas
    # nuevamente podemos aprovechar lo ya calculado.
    pgc_parametros['Desviación estándar'] = math.sqrt(
        pgc_parametros[
            'Varianza'
        ])

    # Imrpimimos la tabla
    encabezado = [
        'Parámetro', 'Valor'
    ]

    print(
        tabulate(pgc_parametros.items(),    # Datos calculados previamente
                 headers=encabezado,        # Encabezados de la tabla
                 tablefmt='presto',         # Hacemos que se vea más bonita
                 floatfmt=".2f")            # Redondeo a 2 decimales
    )


def ejercicioB(df):
    """Item B

    Realice el diagrama tipo serie temporal (valor vs. caso) y
    el tipo constante vs variable.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('B')

    # Convertimos los datos de la columna PGC en una lista
    pgc = df["PGC"].tolist()

    # Lista con el número de caso
    n_pgc = df["N"].tolist()

    # Lista de unos, para el gráfico constante vs valor
    ones = np.ones(len(pgc))

    plot1 = plt.figure(1)
    plt.plot(
        n_pgc,              # Datos eje abscisas
        pgc,                # Datos eje ordenadas
        color='r',          # Color ('r' = red)
        marker='o',         # Marcador ('o' significa "puntos" o "bolas")
        linestyle='none',   # Estilo de línea (no tienen sentido en este caso)
        fillstyle='full',   # Relleno del marcador
        alpha=.3            # Transparencia interior del marcador
    )
    # Título ejes
    plt.xlabel('Caso')
    plt.ylabel('Valor')
    # Título del gráfico
    plt.suptitle('PGC: Grafico valor vs caso')

    plot2 = plt.figure(2)
    plt.plot(
        pgc,              # Datos eje abscisas
        ones,                # Datos eje ordenadas
        color='b',          # Color ('b' = blue)
        marker='o',         # Marcador ('o' significa "puntos" o "bolas")
        linestyle='none',   # Estilo de línea (no tienen sentido en este caso)
        fillstyle='full',   # Relleno del marcador
        alpha=.2            # Transparencia interior del marcador
    )
    # Título ejes
    plt.ylabel('Valor')
    plt.xlabel('Caso')
    # Título del gráfico
    plt.suptitle('PGC: Grafico constante vs valor')

    # Muestro ambos gráficos
    plt.show()


def ejercicioC(df):
    """Item C

    Realice un diagrama boxplot o de caja extraiga alguna conclusión.
    ¿Existen mediciones fuera de lo común o outliers? Haga algún
    comentario sobre los casos outliers. Calcule el porcentaje de
    datos de ese tipo.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('C')

    pgc = df["PGC"].tolist()

    # Realizamos el gráfico (aunque lo mostramos recién al final de la función)
    boxplot_ret = plt.boxplot(
        pgc,                        # Datos
        widths=0.5,                 # Box más ancha
        flierprops=dict(            # Personalizamos los marcadores de outliers
            marker='o',             # Marcador tipo O (círculo)
            markersize=12,          # Tamaño del marcador
            markerfacecolor='b',    # Color del marcador
            fillstyle='full',       # Relleno del marcador
            alpha=.4                # Transparencia interior del marcador
        )
    )
    plt.suptitle('PGC: Boxplot')

    # Calculo outliers manualmente (nobleza obliga).
    # Más abajo se deja comentado cómo obtenerlos directamente a partir del
    # gráfico.

    # Primero ordeno los datos para facilitar el algoritmo que encuentra
    # los bigotes.
    pgc.sort()

    # Obtengo los cuartiles (ver ítem A).
    pgc_quantiles = statistics.quantiles(pgc, n=4)

    # Rango intercuartílico: iqr = q3 - q1
    iqr = pgc_quantiles[2] - pgc_quantiles[0]

    # Límites de Tukey:
    #                   Límite inferior: El menor valor previo a  q1 - 1.5 iqr
    #                   Límite superior: El menor valor previo a  q3 + 1.5 iqr
    lims_tuckey = [pgc_quantiles[0] - 1.5 * iqr, pgc_quantiles[2] + 1.5 * iqr]

    # Bigotes
    # Todos los datos son mayores que la menor de las mediciones,
    # por lo que min(pgc)-1 es un absurdo.
    bigote_inf = bigote_sup = pgc[0] - 1
    # Busco el primer valor superior del límite inferior de Tuckey
    # y el primer valor menor al límite superior de Tuckey.
    for valor in pgc:
        if bigote_inf < pgc[0]-1 and valor > lims_tuckey[0]:
            bigote_inf = valor
        if valor < lims_tuckey[1] and bigote_sup < valor:
            bigote_sup = valor

    # Outliers abajo
    outliers_abajo = valuesBelow(pgc, bigote_inf)
    # Outliers arriba
    outliers_arriba = valuesAbove(pgc, bigote_sup)

    # La proporción de outliers es la razón entre el total de los mismos
    # (la cantidad de outliers por encima + la cantidad de outliers por debajo)
    # y la cantidad de datos obtenidos.
    proporcion_outliers = (len(outliers_arriba) + len(outliers_abajo)) \
        / len(pgc)

    # Imprimimos el porcentaje redondeando a 2 decimales.
    print("Porcentaje de outliers: " +
          str(round(proporcion_outliers * 100, 2)) + "%")

    # Obtención de los outliers a partir de los datos utilizados para graficar
    #
    # Esta parte se encuentra comentada para no imprimir dos veces el mismo
    # resultado.
    #
    # De la documentación de matplotlib, sabemos que boxplot() devuelve un
    # diccionario con los componentes utilizados para la realización del
    # gráfico, entre los cuales se encuentra una lista denominada 'fliers',
    # que contiene los outliers superiores e inferiores.
    #
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html

    # Obteniendo la lista con dichos datos
    fliers = boxplot_ret['fliers'][0].get_data()[1]

    # Simplemente se divide la cantidad de outliers por la cantidad de
    # mediciones.
    proporcion_outliers = len(fliers) / len(pgc)

    # E imprimimos el porcentaje redondeando a 2 decimales.
    print("Porcentaje de outliers: " +
          str(round(proporcion_outliers * 100, 2)) + "%")

    # Mostramos el gráfico
    plt.show()


def ejercicioD(df):
    """Item D

    Realice el histograma y ensaye para elegir el número de intervalos.
    Superponga el polígono de frecuencias sobre el histograma.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('D')

    pgc = df["PGC"].tolist()

    # Para este ítem se superponen dos gráficos.
    fig, ax = plt.subplots()

    # Comienzo creando el histograma
    n, bins, patches = plt.hist(
        pgc,                                  # Datos
        bins=math.ceil(math.log2(len(pgc))),  # Número de intervalos de clase
        color='royalblue',                    # Color del historigrama
        edgecolor='black',                    # Color del borde de los rect.
        linewidth=1,                          # Grosor del borde de los rect.
        label='Histograma'
    )

    # Se obtienen las coordenadas de los puntos con los cuales
    # el polígono de frecuencias es graficado.
    freq_pol_x, freq_pol_y = puntos_poligono_frecuencia(n,
                                                        bins
                                                        )

    # Graficamos el polígono de frecuencias.
    ax.plot(freq_pol_x,                     # Eje abscisas
            freq_pol_y,                     # Eje ordenadas
            linestyle='--',                 # Estilo de línea
            lw=2,                           # Grosor de línea
            color='orange',                 # Color de línea
            marker='h',                     # Marcador ('h' = hexágono)
            markersize=8,                   # Tamaño del marcador
            label='Polígono de frecuencias'
            )

    # Muestro la leyenda con el nombre de cada gráfico
    plt.legend()
    # Muestra el gráfico
    plt.show()


def ejercicioE(df):
    """Item E

    Realice un diagrama de dispersión en donde represente el peso en Kg en
    función de la altura en cm. Hay dos personas cuyo par de datos (altura,
    peso) resultan alejados de la nube de datos.
    Haga algún comentario sobre estos dos casos.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('E')
    peso_lb = df["Peso"].tolist()
    altura_in = df["Altura"].tolist()

    # Convierto los datos de lb a kg y pulgadas a cm, según corresponda
    # Como cada medida es convertida y guardada en el mismo índice, se mantiene
    # la relación entre los datos de ambas listas.
    peso_kg = [libraKilogramo(p) for p in peso_lb]
    altura_cm = [pulgadasCentimetro(a) for a in altura_in]

    plt.plot(
        altura_cm,
        peso_kg,
        'bo'
    )

    # Nombre de los ejes
    plt.xlabel("Altura [cm]")
    plt.ylabel("Peso [kg]")

    # Muestra el gráfico
    plt.show()


def ejercicioF(df):
    """Item F

    El índice de masa corporal IMC se define como la razón entre el peso en Kg
    y el cuadrado de la altura en m. Represente el IMC de esta serie de datos y
    represéntelo en un gráfico tipo serie temporal.
    Calcule el porcentaje de personas para las cuales el IMC es mayor que 25 y
    cuando es mayor que 30.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('F')

    n_datos = df['N'].tolist()
    imc = df["IMC"].tolist()

    # Datos con IMC mayores a 25 y 30 kg/m^2
    above25p = 0
    above30p = 0

    for i in imc:
        if i > 30:          # 30 > 25 -> si n > 30, n > 25
            above30p += 1
            above25p += 1
        elif i > 25:
            above25p += 1

    # Cada cantidad se divide por el largo de la lista con datos sobre el IMC
    above25p /= len(imc)
    above30p /= len(imc)

    print("Personas con un IMC mayor a 25: "
          + str(round(above25p*100, 2)) + "%")
    print("Personas con un IMC mayor a 30: "
          + str(round(above30p*100, 2)) + "%")

    # Gráfico
    plt.plot(
        n_datos,            # Eje de abscisas. Número de orden de cada dato.
        imc,                # Eje de ordenadas.
        color='b',          # Color ('r' = red)
        marker='o',         # Marcador ('o' significa "puntos" o "bolas")
        linestyle='none',   # Estilo de línea (no tienen sentido en este caso)
        fillstyle='full',   # Relleno del marcador
        alpha=.3            # Transparencia interior del marcador
    )

    # Título ejes
    plt.xlabel('Caso')
    plt.ylabel('Valor')
    # Título del gráfico
    plt.suptitle('IMC: Grafico valor vs caso')

    plt.show()


def ejercicioG(df):
    """Item G

    Realice un diagrama de dispersión en donde represente PGC en función del
    IMC.
    ¿Puede extraer alguna conclusión?

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('G')
    pgc = df["PGC"].tolist()
    imc = df["IMC"].tolist()

    plt.plot(
        imc,                # Datos eje abscisas
        pgc,                # Datos eje ordenadas
        color='b',          # Color ('b' = blue)
        marker='o',         # Marcador ('o' significa "puntos" o "bolas")
        linestyle='none',   # Estilo de línea (no tienen sentido en este caso)
        fillstyle='full',   # Relleno del marcador
        alpha=.2            # Transparencia interior del marcador
    )

    # Título ejes
    plt.xlabel("IMC")
    plt.ylabel("PGC")

    plt.show()


def ejercicioH_I(df):
    """Items H e I

    H)
    El coeficiente de correlación ρ es un parámetro que permite analizar el
    grado de dependencia lineal entre los valores medidos de dos variables.
    Si esas variables son independientes entonces el valor de ρ próximo a cero
    mientras que valores cercanos en valor absoluto a 1 dan indicio de
    dependencia lineal y esa situación de dependencia posibilita hacer
    estimaciones de una variable dada un valor de la otra. El cálculo de ρ lo
    realiza el procedimiento corr en Octave. Muestre una tabla con el
    coeficiente ρ, calculado con estos datos, entre PGC y todas las
    restantes variables de las columnas 3 a 17.

    I)
    Con la tabla del ítem anterior elija una de las variables que tenga el
    valor mayor de ρ. Observe que la variable densidad no se recomienda para
    ser elegida ya que para determinarla para una persona en particular habría
    que conocer su volumen, y eso requeriría sumergirla! (y eso es lo que
    quiere evitarse). Una vez elegida la variable entonces realice el diagrama
    de dispersión de PGC en función de esta variable biométrica de fácil
    medición.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'

    Returns:
        string: Nombre de la columna con la variable de menor p respecto del PGC.
    """

    printHeader('H | I')
    pgc = df["PGC"]

    corr_coef = dict()

    for (nombreColumna, datosColumna) in df.iteritems():
        # Evitar columnas N y PGC
        if nombreColumna.lower() == "n" or nombreColumna.lower() == "pgc":
            continue

        # La biblioteca _pandas_ tiene una función llamada corr()
        # Utilizando el metodo de Pearson, es equivalente a corr() en Ocatave
        p = pgc.corr(datosColumna, method='pearson')

        # Guardo el valor de p
        corr_coef[nombreColumna] = p

        # print("Correlación entre PGC y " + nombreColumna + ":  "
        #       + str(round(p, 2)))

    encabezado = [
        'Correlación del PGC', 'p'
    ]
    print(
        tabulate(corr_coef.items(),    # Datos calculados previamente
                 headers=encabezado,        # Encabezados de la tabla
                 tablefmt='presto',         # Hacemos que se vea más bonita
                 floatfmt=".2f")            # Redondeo a 2 decimales
    )

    # Máximo p, despreciando la densidad
    p_maximo = ['', 0]
    for nombre, p in corr_coef.items():
        # Ignoro la densidad
        if (nombre.lower() == "densidad"):
            continue

        if (abs(p) > p_maximo[1]):
            p_maximo[0] = nombre
            p_maximo[1] = abs(p)

    print()  # Línea en blanco para separar de la tabla
    print(
        "El 'p' máximo se obtiene por medio del " + p_maximo[0].lower()
        + ", y su valor absoluto es: "
        + str(round(p_maximo[1], 2))
    )

    # Diagrama de dispersión en función de p_maximo[0] (variable con mayor p).
    plt.plot(
        df[p_maximo[0]],    # Datos eje abscisas
        pgc,                # Datos eje ordenadas
        color='b',          # Color ('b' = blue)
        marker='o',         # Marcador ('o' significa "puntos" o "bolas")
        linestyle='none',   # Estilo de línea (no tienen sentido en este caso)
        fillstyle='full',   # Relleno del marcador
        alpha=.3            # Transparencia interior del marcador
    )

    # Título ejes
    plt.xlabel(p_maximo[0] + ' [cm]')
    plt.ylabel('PGC')

    plt.show()

    return p_maximo[0]


def ejercicioJ(df, columna):
    """Items J

    Estimen PGC para los miembros del grupo que presente este trabajo usando
    la medida de la variable elegida del ítem anterior.
    En Octave/Matlab esta línea de comando realiza esa estimación:
        polyval(polyfit(X,Y,1),a)
    En esta invocación X es un vector con los datos de la variable elegida, Y
    es un vector con los valores de PGC y a es una variable con el valor de la
    variable para la persona de la que quiere estimarse PGC.
    Esa línea de comando obtiene la evaluación del polinomio de grado 1 cuya
    representación gráfica es la denominada recta de mínimos cuadrados que
    mejor ajusta (en ese sentido de los cuadrados mínimos) el diagrama de
    dispersión.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
        columna (string): Encabezado de la columna con la variable de mayor
                          correlación con el PGC.
    """

    printHeader('J')

    pgc = df["PGC"].tolist()

    var_menor_p = df[columna].tolist()

    # TODO: Conseguir datos reales
    mediciones_alumnos = [123, 423, 653, 954, 135]

    # Las funciones polyval y polyfit de Numpy son el reemplazo de las
    # funciones homónimas en Octave/Matlab.
    pgc_alumnos = np.polyval(
        np.polyfit(var_menor_p, pgc, 1),
        mediciones_alumnos
    )

    print(pgc_alumnos)


def ejercicioK(df):
    """---"""

    printHeader('K | L')

    pgc_all = df["PGC"].tolist()
    abdomen_all = df["Abdomen"].tolist()
    imc = df["IMC"].tolist()
    peso_lb = df["Peso"]

    peso_kg = [(n, libraKilogramo(p)) for n, p in peso_lb.iteritems()]

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
    altura_in = df["Altura"].tolist()
    abdomen = df["Abdomen"].tolist()
    cuello = df["Cuello"].tolist()

    altura_cm = [pulgadasCentimetro(a) for a in altura_in]

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
    datos = pd.read_csv('grasacorp.txt', delimiter='\t', header=None)
    datos.columns = COLUMNAS

    print()
    ejercicioA(datos)
    print()  # Línea en blanco
    # ejercicioB(datos)
    # print()
    # ejercicioC(datos)
    # print()
    # ejercicioD(datos)
    # print()
    # ejercicioE(datos)
    # print()
    # ejercicioF(datos)
    # print()
    # ejercicioG(datos)
    # print()
    columna_menor_p = ejercicioH_I(datos)
    # print()
    ejercicioJ(datos, columna_menor_p)
    # print()
    # ejercicioK(datos)
    # print()
    # ejercicioM(datos)
    # print()
