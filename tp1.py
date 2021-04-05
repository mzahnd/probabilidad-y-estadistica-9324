"""Trabajo práctico EAC1

93.24 - Probabilidad y Estadística

Docentes:
.- Villaverde, Francisco
.- Pantazis, Lucio
.- Cosatto, Pedro

Alumnos:
.- Carolo, Lorena
.- Pardiñas, Victoria
.- Peydro, Florencia
.- Scalise, Zarina
.- Zahnd, Martín E.

Fuentes y documentación consultadas:
    Documentación de matplotlib
    https://matplotlib.org/stable/tutorials/index.html
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
    https://matplotlib.org/stable/gallery/color/named_colors.html

    Documentación de tabulate
    https://pypi.org/project/tabulate/

    Documentación del paquete statistics
    https://docs.python.org/3/library/statistics.html

    Ejemplos de Pandas
    https://www.geeksforgeeks.org/loop-or-iterate-over-all-or-certain-columns-of-a-dataframe-in-python-pandas/

    RealPython
    https://realpython.com/numpy-scipy-pandas-correlation-python/


Licencia:
MIT License

Copyright (c) 2021 Martín E. Zahnd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import copy
import math
import statistics
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt


# Columnas del archivo con datos
COLUMNAS = [
    'N', 'PGC', 'Densidad', 'Edad', 'Peso', 'Altura',
    'IMC', 'Cuello', 'Pecho', 'Abdomen', 'Cadera',
    'Muslo', 'Rodilla', 'Tobillo', 'Biceps', 'Antebrazo',
    'Muñeca'
]

# Índice para crear las figuras
FIGURE_INDEX = 0


def printHeader(letra):
    """Imprime un encabezado para el ejercicio con texto: "Ejercicio L".

    Args:
        letra (str): Letra del o los ejercicios,
    """
    if (len(letra) > 1):
        print("Ejercicios " + letra)
    else:
        print("Ejercicio " + letra)
    print("="*30)
    print()


def printTable(encabezado, datos):
    """Imprime una tabla utilizando tabulate, con dos decimales.

    Args:
        encabezado (list): Títulos para los encabezados
        datos (list / dict): Contenido de la tabla
    """
    print(
        tabulate(datos,                     # Datos para completar la tabla
                 headers=encabezado,        # Encabezados de la tabla
                 tablefmt='presto',         # Hacemos que se vea más bonita
                 floatfmt=".2f")            # Redondeo a 2 decimales
    )


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
    """Convierte de libras a kg.
    """

    return pounds*0.45359237


def pulgadaCentimetro(inches):
    """Convierte de pulgadas a centímetros.
    """

    return inches*2.54


def puntos_poligono_frecuencia(bins_val, bins):
    """Calcula los puntos X e Y para crear el polígono de frecuencias.

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


def diagrama_puntos(datos_abs, datos_ord, marker_color='blue',
                    marker_style='o', marker_alpha=0.3, marker_size=9,
                    titulo_ejes=['', ''], titulo_figura='', nombre_leyenda='',
                    mostrar_leyenda=False, crear_figure=True,
                    titulo_ejes_size=14, titulo_figura_size=16,
                    leyenda_size=12):
    """Crear un diagrama con puntos.

    Crea una plt.figure e incrementa el índice (FIGURE_INDEX) automáticamente.

    Args:
        datos_abs (list): Datos para el eje de abscisas
        datos_ord (list): Datos para el eje de ordenadas
        marker_color (str, optional): Color del marcador. Por defecto: 'blue'.
        marker_style (str, optional): Estilo del marcador. Por defecto: 'o'.
        marker_alpha (float, optional): Transparencia del marcador.
        Por defecto: 0.3.
        titulo_ejes (list, optional): Título para el eje X e Y. Por defecto: ''
        titulo_figura (str, optional): Título del gráfico. Por defecto: ''
        nombre_leyenda (str, optional): Nombre para la leyenda. Por defecto: ''
        marker_size (int, optional): Tamaño del marcador. Por defecto: 9
        mostrar_leyenda (bool, optional): Mostrar leyenda. Por defecto: False
        crear_figure (bool, optional): Crea una plt.figure y aumenta el
        FIGURE_INDEX automáticamente. Por defecto: True
        titulo_ejes_size (int, optional): Tamaño de fuente de los ejes.
        Por defecto: 14
        titulo_figura_size (int, optional): Tamaño de fuente del título.
        Por defecto: 16
        leyenda_size (int, optional): Tamaño de fuente del texto en la leyenda.
        Por defecto: 12
    """

    if (crear_figure):
        global FIGURE_INDEX
        FIGURE_INDEX += 1

        fig = plt.figure(FIGURE_INDEX)

    plt.plot(
        datos_abs,                  # Datos eje abscisas
        datos_ord,                  # Datos eje ordenadas
        color=marker_color,         # Color
        marker=marker_style,        # Marcador
        markersize=marker_size,     # Tamaño del marcador
        alpha=marker_alpha,         # Transparencia interior del marcador
        fillstyle='full',           # Relleno del marcador
        linestyle='none',           # Estilo de línea (no tiene sentido)
        label=nombre_leyenda,
    )

    # Mostrar leyenda
    if (mostrar_leyenda):
        plt.legend(fontsize=leyenda_size)

    # Nombre ejes
    plt.xlabel(titulo_ejes[0], fontsize=titulo_ejes_size)
    plt.ylabel(titulo_ejes[1], fontsize=titulo_ejes_size)

    # Título gráfico
    plt.suptitle(titulo_figura, fontsize=titulo_figura_size)

    # No desperdiciemos espacio en la ventana del gráfico
    plt.tight_layout()


def estimarPGC(h, a, n, sexo):
    """Estimación del PGC utilizando la fórmula del item M.

    Args:
        h (float): Altura en centímetros
        a (float): Circunferencia abdominal en centímetros
        n (float): Circunferencia del cuello en centímetros
        sexo (string): 'M': Masculino ; 'F': Femenino

    Returns:
        float: PGC estimado
    """

    est = 0
    if sexo == 'M':
        est = 10.1 - 0.239 * h + 0.8 * a - 0.5 * n
    elif sexo == 'F':
        est = 19.2 - 0.239 * h + 0.8 * a - 0.5 * n

    return est


def ejercicio_A(df):
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

    printTable(encabezado, pgc_parametros.items())


def ejercicio_B(df):
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

    # Valor vs caso
    diagrama_puntos(
        n_pgc,
        pgc,
        titulo_figura='PGC: Grafico valor vs caso',
        titulo_ejes=['Caso', 'Valor']
    )

    # Valor  vs cte
    diagrama_puntos(
        pgc,
        ones,
        titulo_figura='PGC: Grafico constante vs valor',
        titulo_ejes=['Valor', 'Constante']
    )


def ejercicio_C(df):
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
    global FIGURE_INDEX
    FIGURE_INDEX += 1
    plot_C = plt.figure(FIGURE_INDEX)
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


def ejercicio_D(df):
    """Item D

    Realice el histograma y ensaye para elegir el número de intervalos.
    Superponga el polígono de frecuencias sobre el histograma.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
    """

    printHeader('D')

    pgc = df["PGC"].tolist()

    global FIGURE_INDEX
    FIGURE_INDEX += 1
    plot_D = plt.figure(FIGURE_INDEX)

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
    plt.plot(freq_pol_x,                     # Eje abscisas
             freq_pol_y,                     # Eje ordenadas
             linestyle='--',                 # Estilo de línea
             lw=2,                           # Grosor de línea
             color='orange',                 # Color de línea
             marker='h',                     # Marcador ('h' = hexágono)
             markersize=8,                   # Tamaño del marcador
             label='Polígono de frecuencias'
             )

    # Muestro la leyenda con el nombre de cada gráfico
    plt.legend(fontsize=12)


def ejercicio_E(df):
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
    altura_cm = [pulgadaCentimetro(a) for a in altura_in]

    # Cálculos extra para el informe escrito.
    # Se crea un diccionario que contiene la media, mediana, máximo y mínimo de
    # cada conjunto de datos, y luego se imprime una tabla con los mismos.
    extra = dict()
    extra[''] = ['Peso [kg]', 'Altura [cm]']

    # Media
    extra['Media muestral'] = [
        statistics.mean(peso_kg),
        statistics.mean(altura_cm)
    ]

    # Mediana
    extra['Mediana'] = [
        statistics.median(peso_kg),
        statistics.median(altura_cm)
    ]

    # Max,min
    extra['Máximo'] = [max(peso_kg), max(altura_cm)]
    extra['Mínimo'] = [min(peso_kg), min(altura_cm)]

    # Tabla
    encabezado = [key for key, _ in extra.items()]
    printTable(encabezado, extra)

    # Gráfico
    diagrama_puntos(
        altura_cm,
        peso_kg,
        titulo_figura='Relación altura-peso',
        titulo_ejes=['Altura [cm]', 'Peso [kg]'],
    )


def ejercicio_F(df):
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
    diagrama_puntos(
        n_datos,
        imc,
        titulo_figura='IMC: Gráfico valor vs caso',
        titulo_ejes=['Caso', 'Valor'],
    )


def ejercicio_G(df):
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

    # Gráfico
    diagrama_puntos(
        imc,
        pgc,
        titulo_figura='Diagrama de dispersión',
        titulo_ejes=['IMC', 'PGC'],
    )


def ejercicio_H_I(df):
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
        string: Nombre de la columna con la variable de menor p respecto de PGC
    """

    printHeader('H e I')
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

    # Imrpimimos la tabla
    encabezado = [
        'Correlación del PGC', 'p'
    ]
    printTable(encabezado, corr_coef.items())

    # Máximo p, despreciando la densidad
    p_maximo = ['', 0]
    for nombre, p in corr_coef.items():
        # Ignoro la densidad
        if (nombre.lower() == "densidad"):
            continue

        if (abs(p) > p_maximo[1]):
            p_maximo[0] = nombre
            p_maximo[1] = abs(p)

    print()  # Línea en blanco para separar de la tabla del título
    print(
        "El 'p' máximo se obtiene por medio del " + p_maximo[0].lower()
        + ", y su valor absoluto es: "
        + str(round(p_maximo[1], 2))
    )

    # Diagrama de dispersión en función de p_maximo[0] (variable con mayor p).
    diagrama_puntos(
        df[p_maximo[0]],
        pgc,
        titulo_figura='Diagrama de dispersión',
        titulo_ejes=[p_maximo[0] + ' [cm]', 'PGC'],
    )

    return p_maximo[0]


def ejercicio_J(df, columna):
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

    mediciones_alumnos = [84, 85, 70, 80, 82]

    # Las funciones polyval y polyfit de Numpy son el reemplazo de las
    # funciones homónimas en Octave/Matlab.
    pgc_alumnos = np.polyval(
        np.polyfit(var_menor_p, pgc, 1),
        mediciones_alumnos
    )

    # Para la tabla
    # Una lista con cada elemento conteniendo otra lista de forma
    # [i, pgc] , donde i es un número natural.
    pgc_alumnos_tabla = list()
    for i in range(len(mediciones_alumnos)):
        pgc_alumnos_tabla.append([i+1, pgc_alumnos[i]])

    encabezado = ['Alumno', 'PGC estimado']
    printTable(encabezado, pgc_alumnos_tabla)


def ejercicio_K_L(df, columna):
    """Items K y L

    K)
    Divida la población en dos grupos de acuerdo con el peso entre los que
    pesan menos de 70 Kg y los de peso igual o mayor. En cada grupo determine
    el porcentaje de los que tienen IMC mayor que 25.

    L)
    Realice dos diagramas de dispersión superpuestos donde se represente PGC
    en función de la variable elegida en el ítem i para los dos grupos del
    ítem anterior. Represente cada serie con un símbolo diferente. Haga algún
    comentario de lo que observe.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
        columna (string): Encabezado de la columna con la variable de mayor
                          correlación con el PGC.
    """

    printHeader('K y L')

    # Datos
    pgc_all = df["PGC"].tolist()
    variable_elegida_all = df[columna].tolist()
    imc = df["IMC"].tolist()
    peso_lb = df["Peso"]

    # Convertimos el peso a kilogramos, manteniendo el número de orden (n)
    peso_kg = [(n, libraKilogramo(p)) for n, p in peso_lb.iteritems()]

    datos_grupo = {
        'pgc': list(),               # PGC de cada individuo del grupo
        'var': list(),               # Variable elegida
        'imc>25': [int(), float()],  # Cant y % de individuos con IMC > 25
    }

    peso_por_grupo = {
        # datos_grupo es un template para crear este diccionario.
        # Lo copiamos para que ambos elementos no hagan referencia a la misma
        # porción en memoria (cosas de Python)
        '< 70': copy.deepcopy(datos_grupo),
        '>= 70': copy.deepcopy(datos_grupo),
    }

    for n, w in peso_kg:
        if w < 70:
            grupo = '< 70'
        else:
            grupo = '>= 70'

        peso_por_grupo[grupo]['pgc'].append(pgc_all[n])
        peso_por_grupo[grupo]['var'].append(variable_elegida_all[n])
        if (pgc_all[n] > 25):
            peso_por_grupo[grupo]['imc>25'][0] += 1

    # Se calcula el porcentaje de individuos con IMC > 25, en cada grupo
    for grupo, _ in peso_por_grupo.items():
        peso_por_grupo[grupo]['imc>25'][1] = \
            peso_por_grupo[grupo]['imc>25'][0] \
            / len(peso_por_grupo[grupo]['pgc'])

    # Preparamos e imprimomos una tabla
    encabezado = [
        'Peso [kg]',
        'Personas en\neste grupo',
        'Personas con\nIMC mayor a 25',
        'Proporción [%]'
    ]

    datos_tabla = list()
    for grupo, _ in peso_por_grupo.items():
        datos_tabla.append(
            [
                grupo,
                len(peso_por_grupo[grupo]['pgc']),
                peso_por_grupo[grupo]['imc>25'][0],
                peso_por_grupo[grupo]['imc>25'][1] * 100
            ]
        )

    printTable(encabezado, datos_tabla)

    # Gráficos
    # Para crear dos gráficos superpuestos, en el primero se crea una
    # plt.figure y en el segundo no, de forma tal que el segundo plt.plot se
    # haga sobre la misma figura que el primero.
    diagrama_puntos(
        peso_por_grupo['< 70']['var'],
        peso_por_grupo['< 70']['pgc'],
        nombre_leyenda='Peso menor a 70 kg',
        crear_figure=True,
    )

    diagrama_puntos(
        peso_por_grupo['>= 70']['var'],
        peso_por_grupo['>= 70']['pgc'],
        marker_color='g',
        marker_style='*',
        nombre_leyenda='Peso mayor o igual a 70 kg',
        titulo_figura='Diagramas de dispersión',
        titulo_ejes=[columna + ' [cm]', 'PGC'],
        mostrar_leyenda=True,
        crear_figure=False
    )


def ejercicio_M(df):
    """Item M

    En un artículo publicado en British Journal of Nutrition, un grupo de
    científicos de Israel desarrolló distintas formulas para una estimación
    rápida y económica del PGC. Las formulas que se muestran a continuación
    estiman el PGC a partir de tres medidas en cm: la altura H, la circunferen-
    cia abdominal A y la circunferencia del cuello N.
        PGC = 10.1 – 0.239 H + 0.8 A – 0.5 N        (para hombres)
        PGC = 19.2 – 0.239 H + 0.8 A – 0.5 N        (para mujeres)

    Realice un gráfico superponiendo los datos de PGC de la tabla y los
    calculados con estas fórmulas y comente lo observado.

    Args:
        df (pandas data frame): Conjunto de datos del archivo 'grasacorp.txt'
        columna (string): Encabezado de la columna con la variable de mayor
                          correlación con el PGC.
    """

    printHeader('M')

    # Datos
    pgc = df["PGC"]

    altura_in = df["Altura"].tolist()
    abdomen = df["Abdomen"].tolist()
    cuello = df["Cuello"].tolist()

    # Convertimos la altura a centímetros
    altura_cm = [pulgadaCentimetro(a) for a in altura_in]

    # Desconozco el sexo de los individuos. Estimo el PGC con ambos sexos.
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

    # Creamos dos listas, una con el número de orden de cada individuo, la
    # segunda con el pgc medido de cada individuo.
    n_pgc = list()
    pgc_val = list()
    for n, pgc in pgc.iteritems():
        n_pgc.append(n)
        pgc_val.append(pgc)

    # Para crear tres gráficos superpuestos, en el primero se crea una
    # plt.figure y en el resto no, de forma tal que el segundo y tercer
    # plt.plot se hacen sobre la misma figura que el primero.
    diagrama_puntos(
        n_pgc,
        pgc_val,
        marker_color='forestgreen',
        marker_alpha=0.2,
        nombre_leyenda='PGC Medido',
        crear_figure=True,
    )

    diagrama_puntos(
        n_pgc,
        pgc_est_hombres,
        marker_color='blue',
        marker_style='^',
        marker_alpha=0.4,
        nombre_leyenda='PGC estimado: Población masculina',
        crear_figure=False,
    )

    diagrama_puntos(
        n_pgc,
        pgc_est_mujeres,
        marker_color='fuchsia',
        marker_style='*',
        marker_alpha=0.4,
        nombre_leyenda='PGC estimado: Población femenina',
        titulo_figura='Comparación del PGC medido con fórmula de estimación',
        titulo_ejes=['Número de orden', 'PGC'],
        mostrar_leyenda=True,
        crear_figure=False
    )


if __name__ == "__main__":
    # Obtención de los datos
    datos = pd.read_csv('grasacorp.txt', delimiter='\t', header=None)

    # Ponemos nombre a cada columna
    datos.columns = COLUMNAS

    # Ejercicios
    print()
    ejercicio_A(datos)
    print()  # Línea en blanco
    ejercicio_B(datos)
    print()
    ejercicio_C(datos)
    print()
    ejercicio_D(datos)
    print()
    ejercicio_E(datos)
    print()
    ejercicio_F(datos)
    print()
    ejercicio_G(datos)
    print()
    columna_menor_p = ejercicio_H_I(datos)
    print()
    ejercicio_J(datos, columna_menor_p)
    print()
    ejercicio_K_L(datos, columna_menor_p)
    print()
    ejercicio_M(datos)
    print()

    # Mostramos todos los gráficos
    # plt.show()

    plt.close()
