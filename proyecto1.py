""""

PROYECTO 1  Aplicaciones de IA

Aprendizaje Supervisado


Nombre: Armando Rodriguez


email: armandospxp@gmail.com


"""


###Importacion de librerias

# librerias de tipos
from typing import Tuple

# pandas y matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# dataset
from sklearn.datasets import load_breast_cancer

# modelos ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# standarizacion numerica y split de datos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# metricas
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score


def carga_datos()->Tuple[pd.DataFrame, pd.DataFrame]:
    """Metodo que retorna 2 dataframes con la data necesaria para el entrenamiento"""
    data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.DataFrame(data.target, columns=['target'])


def limpieza_estandarizacion(X:pd.DataFrame, y:pd.DataFrame, size:float=0.2)-> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Metodo para realizar limpieza y estandarizacion del dataframe,
    recibe como parametros un dataframe y un size para la distribucion, 0.3 para 70-30, 0.2 para 80-20
    por default es 0.2
    retorna una tupla con los train test split normalizados"""
    # Escala de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=size)

if __name__ == '__main__':
    """Corrida del script"""
    print("Inicio del script")
    print("Carga de datos")
    df, df_target = carga_datos()
    print("Estandarizacion de los datos")
    X_train, X_test, y_train, y_test= limpieza_estandarizacion(df, df_target, size=0.3)
    # Cargamos los modelos en un diccionario para mejor tratamiento
    modelos = {'random_forest':RandomForestClassifier(random_state=42), 'svm':SVC(probability=True, random_state=42)}

    # almacenamiento de resultados(estaremos iterando y guardando en un diccionario el resultado de los entrenamientos
    # y pruebas)

    resultados = {}

    # Comenzamos a iterar para entrenar los modelos
    print("Comienza entrenamiento de los datos")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

       # medicion de prediccion

        y_pred = modelo.predict(X_test)
        # prediccion de probabilidad (la probabilidad genera un array de 2 dimensiones,
        # se agarra la dimension 1 que es la probabilidad de la prediccion de la variable objetivo)

        y_proba = modelo.predict_proba(X_test)[:, 1]

        # calculo de las metricas

        metricas = {
           'accuracy':accuracy_score(y_pred, y_test),
           'roc_auc':roc_auc_score(y_test, y_proba),
           'f1_score':f1_score(y_pred, y_test)
        }
        resultados[nombre] = metricas

    resultados_df = pd.DataFrame(resultados).T

    print("Metricas de desempenio de RandomForest vs SVM:")
    print(resultados_df)

    # Grafico de curvas ROC

    plt.figure(figsize=(8, 6))
    for nombre, modelo in modelos.items():
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Aleatorio (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC: RandomForest vs SVM')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
