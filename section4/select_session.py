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

"""
**********
SECTION 3 - AJUSTANDO MODELOS
**********

DIVIDIENDO LOS DATOS ENTRE ENTRENAMIENTO Y PRUEBA
Para comprender si el modelo realmente está aprendiendo de los datos,
    necesitamos hacer una separación de los datos entre entrenamiento
    y prueba. Los datos de entrenamiento se utilizan para ajustar el
    modelo, mientras que los datos de prueba sirve para verificar el
    aprendizaje del modelo en datos que no fueron utilizados en el
    momento del ajuste.
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25,
                                                    random_state=5,
                                                    stratify=y)

"""
MODELO DE REFERENCIA - BASELINE
El modelo más simple de clasificar los datos es simplemente utilizar un
    algoritmo que asigna todas las clasificaciones a la clase que tiene
    mayor frecuencia. Este algoritmo sirve como un criterio de
    comparación para identificar si los otros modelos tienen un
    rendimiento mejor que la clasificación más simple posible.
"""
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
dummy.fit(X_train, y_train)

dummy.score(X_test, y_test)

"""
ARBOLES DE DECISIÓN
    ¬ Explicabilidad y procesamiento

    ¬ El algoritmo segmenta los datos comparando entre menor y mayor

It does this analysis with every column
    A   B   C<=13       R
    10  5   31          1 X (BAD)
    30  8   14          0
    25  10  12          1
    15  9   7           1

El índice de GINI calcula qué tan mezcladas están las clases. Varía
    de 0 a 1, dónde 0 es perfectamente separado, y 1 completamente
    mezclado.

El modelo de árbol de decisión es muy utilizado debido a su alta
    explicabilidad y procesamiento rápido, manteniendo un rendimiento
    bastante interesante.

Se basa en decisiones simples por el algoritmo, separando los datos
    mediante comparaciones de menor y mayor en los valores de las
    columnas de la base de datos.
"""
from sklearn.tree import DecisionTreeClassifier

modelo_arbol = DecisionTreeClassifier(random_state=5)
modelo_arbol.fit(X_train, y_train)
modelo_arbol.score(X_test, y_test)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

valores_columnas = ["casado (a)",
                    "divorciado (a)",
                    "soltero (a)",
                    "primaria",
                    "secundaria",
                    "superior",
                    "default",
                    "prestatario",
                    "edad",
                    "saldo",
                    "ultimo_contacto",
                    "ct_contactos"]

plt.figure(figsize=(80, 25))
plot_tree(modelo_arbol, filled=True, 
          class_names=["no", "si"], 
          fontsize=6, feature_names=valores_columnas)

# Prueba de overfitting 
modelo_arbol.score(X_train, y_train)    # Para los datos de entrenamiento

# Segundo modelo de arbol de decisiones
modelo_arbol2 = DecisionTreeClassifier(max_depth=3,
                                      random_state=5)
modelo_arbol2.fit(X_train, y_train)
modelo_arbol2.score(X_test, y_test)
plt.figure(figsize=(15, 6))
plot_tree(modelo_arbol2, filled=True, 
          class_names=["no", "si"], 
          fontsize=6, feature_names=valores_columnas)

"""
**********
SECTION 4 - SELECCIÓN DE MODELOS
**********

NORMALIZANDO LOS DATOS
Algunos algoritmos pueden asignar un mayor peso a los valores de las
    variables debido a la escala de los valores y no por la importancia
    de la clasificación de la variable objetivo. Por ejemplo, en una
    base de datos con las columnas edad y saldo, el algoritmo puede dar
    un mayor peso de decisión a los valores del saldo simplemente por
    estar en una escala mayor que los valores de la edad, y no porque
    la variable saldo sea más importante que la variable edad.

En estos casos, necesitamos realizar una transformación en los datos
    para que estén en una misma escala, evitando que el algoritmo sea
    influenciado incorrectamente por los valores numéricos divergentes
    entre las variables.
"""
from sklearn.preprocessing import MinMaxScaler

normalizacion = MinMaxScaler()
X_train_normalizada = normalizacion.fit_transform(X_train)

pd.DataFrame(X_train_normalizada)

"""
KNN
    ¬ Identifica los vecinos más cercanos al punto de consulta (variable
        obejtivo o de respuesta).
    
    ¬ Clasifica el punto de consuta basado en los valores vecinos.

El algoritmo KNN se basa en el cálculo de la distancia entre los
    registros de la base de datos y busca elementos que estén cerca
    unos de otros (vecinos) para tomar la decisión de clasificación.

Debido a que utiliza cálculos de distancia, este algoritmo está
    influenciado por la escala de las variables, y por eso es necesario
    realizar una transformación en los datos antes de utilizar este
    método.
"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_normalizada, y_train)
X_test_normalizado = normalizacion.transform(X_test)
knn.score(X_test_normalizado, y_test)

"""
ESCOGIENDO Y SERIALIZANDO EL MEJOR MODELO
Al final de un proyecto de machine learning, debemos comparar los
    resultados de los modelos y elegir el que tenga el mejor
    rendimiento.

Podemos almacenar el modelo en un archivo serializado del tipo pickle
    para que sea utilizado en producción, es decir, en datos del mundo
    real para atender las necesidades del problema que necesita ser
    resuelto.
"""
lista = [("dummy", dummy, X_test),
         ("de árbol", modelo_arbol2, X_test),
         ("knn", knn, X_test_normalizado)]

for i in lista:
    print(f"La exactitud del modelo {i[0]}: {i[1].score(i[2], y_test)}")

import pickle

with open("modelo_onehotencoder.pkl", "wb") as archivo:
    pickle.dump(one_hot, archivo)

with open("modelo_champion.pkl", "wb") as archivo:
    pickle.dump(modelo_arbol2, archivo)

nuevo_dato = {"edad": [45],
              "estado_civil": ["soltero (a)"],
              "escolaridad": ["superior"],
              "default": ["no"],
              "saldo": [23040],
              "prestatario": ["no"],
              "ultimo_contacto": [800],
              "ct_contactos": [4]}

nuevo_dato = pd.DataFrame(nuevo_dato)
nuevo_dato

modelo_one_hot = pd.read_pickle("../models/modelo_onehotencoder.pkl")
modelo_champion = pd.read_pickle("../models/modelo_champion.pkl") # Arbol 2

# Esto da error porque no se ha pasado por el onehot
# modelo_arbol2.predict(nuevo_dato)

nuevo_dato = modelo_one_hot.transform(nuevo_dato)
result = modelo_champion.predict(nuevo_dato)
result
print(type(result))
