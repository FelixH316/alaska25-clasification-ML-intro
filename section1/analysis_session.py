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
