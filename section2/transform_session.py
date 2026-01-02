"""
CLASIFICACIÓN: PRIMEROS PASOS
En este proyecto, analizaremos datos de una campaña de marketing para
    la adhesión a inversiones. El objetivo es utilizar la información
    de los datos para predecir si los clientes de un banco invertirán
    su dinero o no.

Esta predicción se realizará mediante machine learning, y este notebook
    contendrá los pasos para obtener un modelo capaz de hacer dichas
    predicciones, desde la lectura, análisis exploratorio, separación y
    transformación de los datos, hasta el ajuste, evaluación y 
    comparación de modelos de clasificación.
"""

"""
**********
SECTION 1 - ANÁLISIS EXPLORATORIO
**********

REALIZANDO LA LECTURA DE LOS DATOS
Podemos leer los datos ulizando la biblioteca pandas. Por esta razón
    importaremos la biblioteca con el comando import pandas as pd

Dado que el archivo de datos está en formato csv, realizaremos una
    lectura con la función read_csv()
"""
import pandas as pd

datos = pd.read_csv("../datasets/marketing_inversiones.csv")
datos.head(5)

# Para crear modelos de clasificación, necesitamos utilizar datos de
#   calidad, sin inconsistencias y sin datos faltantes. Verificaremos
#   si existen datos nulos y el tipo de cada columna en la base de
#   base de datos utilizando el método info()
datos.info()

"""
EXPLORANDO LOS DATOS
Una etapa muy importante en proyectos de machine learning es la
    exploración y comprensión de los datos, conocida como análisis
    exploratorio. Podemos utilizar gráficos para verificar que
    información contiene cada una de las columnas de la base de datos,
    así como identificar inconsistencias y patrones que puedan existir.

Exploraremos cada una de las columnas de la base de datos utilizando la
    biblioteca plotly. Comenzaremos con las variables categóricas y
    luego analizaremos las variable numéricas.
"""
# Variables categóricas
import plotly.express as px

px.histogram(datos, x="adherencia_inversion", text_auto=True)

px.histogram(datos, x="estado_civil", text_auto=True, color="adherencia_inversion", barmode="group")

px.histogram(datos, x="escolaridad", text_auto=True, color="adherencia_inversion", barmode="group")

px.histogram(datos, x="default", text_auto=True, color="adherencia_inversion", barmode="group")

px.histogram(datos, x="prestatario", text_auto=True, color="adherencia_inversion", barmode="group")

# Variables númericas
px.box(datos, x="edad", color="adherencia_inversion")

px.box(datos, x="saldo", color="adherencia_inversion")

px.box(datos, x="ultimo_contacto", color="adherencia_inversion")

px.box(datos, x="ct_contactos", color="adherencia_inversion")

"""
**********
SECTION 2 - TRANSFORMACIÓN DE LOS DATOS
**********

VARIABLES EXPLICATIVAS Y VARIABLE DE RESPUESTA
Para realizar la predicción de los valores con un modelo de apredizaje
    automático, necesitamos separar la variable objetivo de las
    variables explicativas. La variable y representa lo que queremos
    predecir, mientras que x incluye todas las variables que se
    utilizarán para explicar el comportamiento de y.
"""
datos
X = datos.drop("adherencia_inversion", axis=1)
y = datos["adherencia_inversion"]

X   # Pandas array (variables explicativas)
y   # Pandas series (variable objetivo)

"""
TRANSFORMANDO LAS VARIABLES EXPLICATIVAS
Los algoritmos de aprendizaje automático no comprenden datos en formato
    de texto, por lo que debemos transformar los datos a un formato
    númerico para que el algoritmo pueda interpretar la información.

Esta transformación debe realizarse de manera que no altere la
    información original del conjunto de datos, por lo que no basta con
    simplemente cambiar los valores a números aleatorios.

        A   B   C   D
A ->    1   0   0   0
B ->    0   1   0   0
A ->    1   0   0   0
D ->    0   0   0   1
C ->    0   0   1   0
"""
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

columnas = X.columns
one_hot = make_column_transformer((OneHotEncoder(drop="if_binary"), # Ignora columnas binarias
                                  ["estado_civil",
                                   "escolaridad",
                                   "default",
                                   "prestatario"]),
                                  remainder="passthrough",          # Que hacer con las columnas restantes, omitir
                                  sparse_threshold=0,               # No quitar información relevante
                                  force_int_remainder_cols=False)   # Hace obligatorio que se cambie a int el nombre de la columna
                                                                    #   A partir de v1.9 por default será False y se deprecará

# Variables explicativas
X = one_hot.fit_transform(X)
one_hot.get_feature_names_out(columnas)

X
pd.DataFrame(X, columns=one_hot.get_feature_names_out(columnas))

"""
TRANSFORMANDO LA VARIABLE RESPUESTA
Así como las variables explicativas, la varible objetivo también debe
    convertirse al formato numérico. Podemos representar una variable
    objetivo binaria como 0 o 1, donde 0 indica la ausencia de la
    característica de la variable y 1 representa su presencia.
"""
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
y


