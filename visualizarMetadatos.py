import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

LONGITUD_EJEMPLO = 131
PATH_VALORES = 'X_train.csv'
PATH_ETIQUETAS = 'Y_train.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosidad", "-v", type=bool, default=False, help="Booleano para mostrar informacion detallada, por defecto: False")
    parser.add_argument("--path_valores_personalizado", "-pv", type=str, default=PATH_VALORES, help="Ruta del archivo de valores, por defecto: " + PATH_VALORES)
    parser.add_argument("--path_etiquetas_personalizado", "-pe", type=str, default=PATH_ETIQUETAS, help="Ruta del archivo de etiquetas, por defecto: " + PATH_ETIQUETAS)
    parser.add_argument("--longitud_ejemplo", "-l", type=int, default=LONGITUD_EJEMPLO, help="Longitud del dato de entrenamiento, por defecto: " + str(LONGITUD_EJEMPLO))
    parser.add_argument("--no_mostrar_ejemplo", "-nm", action='store_false', dest='mostrar_ejemplo', help="Flag para no mostrar el ejemplo")
    parser.add_argument("--guardar_ejemplo", "-g", action='store_true', help="Flag para guardar el ejemplo")
    args = parser.parse_args()

    df_valores = pd.read_csv(args.path_valores_personalizado, delimiter=';')
    df_etiquetas = pd.read_csv(args.path_etiquetas_personalizado, delimiter=';')

    unit_scaler = MinMaxScaler().set_output(transform="pandas")
    unit_scaler.fit(df_valores)
    df_valores_scl = unit_scaler.transform(df_valores)

    if args.verbosidad:
        print("Header Valores")
        print(df_valores_scl.head())

        print("Header Etiquetas")
        print(df_etiquetas.head())

        print("Shape Valores")
        print(df_valores_scl.shape)

        print("Shape Etiquetas")
        print(df_etiquetas.shape)

    # Visualizar la cantidad de cada clase
    print("Cantidad de cada clase:")
    print(df_etiquetas.iloc[:, 1].value_counts())
    
    # Visualizar los porcentajes de cada clase
    print("Porcentaje de cada clase:")
    print(df_etiquetas.iloc[:, 1].value_counts(normalize=True)*100)

    # Calcular la varianza por columna de cada clase, omitiendo la primera columna (ID)
    varianza_por_clase = df_valores_scl.iloc[:, 1:].groupby(df_etiquetas.iloc[1:, 1]).var()
    print("Varianza por columna de cada clase:")
    print(varianza_por_clase)

    # Verificar el contenido de varianza_por_clase
    print(varianza_por_clase)

    # Asegúrate de que varianza_por_clase es un DataFrame
    if isinstance(varianza_por_clase, pd.DataFrame):
        # Calcular la varianza total de cada columna
        varianza_total = df_valores_scl.iloc[:, 1:].var()

        plt.figure(figsize=(12, 6))
        ax = plt.gca()  # Obtener el objeto Axes actual
        varianza_por_clase.T.plot(kind='bar', ax=ax, stacked=True, colormap='tab20', alpha=0.7, label='Varianza por clase')  # Cambiar a 'tab20'
        varianza_total.plot(kind='line', ax=ax, color='black', linewidth=2, marker='o', label='Varianza total')  # Añadir varianza total
        plt.title('Varianza por columna para cada clase y total')
        plt.xlabel('Columnas')
        plt.ylabel('Varianza')
        plt.xticks(rotation=45)
        plt.legend(title='Clases')
        plt.tight_layout()
        plt.show()
    else:
        print("varianza_por_clase no es un DataFrame")
