import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import load_data, scale_data

LONGITUD_EJEMPLO = 131
PATH_VALORES = 'X_train.csv'
PATH_ETIQUETAS = 'Y_train.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indice", "-i", type=int, default=0, help="Indice del ejemplo a visualizar")
    parser.add_argument("--verbosidad", "-v", type=bool, default=False, help="Booleano para mostrar informacion detallada, por defecto: False")
    parser.add_argument("--path_valores_personalizado", "-pv", type=str, default=PATH_VALORES, help="Ruta del archivo de valores, por defecto: " + PATH_VALORES)
    parser.add_argument("--path_etiquetas_personalizado", "-pe", type=str, default=PATH_ETIQUETAS, help="Ruta del archivo de etiquetas, por defecto: " + PATH_ETIQUETAS)
    parser.add_argument("--longitud_ejemplo", "-l", type=int, default=LONGITUD_EJEMPLO, help="Longitud del dato de entrenamiento, por defecto: " + str(LONGITUD_EJEMPLO))
    parser.add_argument("--no_mostrar_ejemplo", "-nm", action='store_false', dest='mostrar_ejemplo', help="Flag para no mostrar el ejemplo")
    parser.add_argument("--guardar_ejemplo", "-g", action='store_true', help="Flag para guardar el ejemplo")
    parser.add_argument("--cantidad", "-c", type=int, default=1, help="Cantidad de ejemplos a visualizar a partir del índice dado")
    parser.add_argument("--same_class", "-sc", action='store_true', help="Flag para que todos los ejemplos sean de la misma clase")
    args = parser.parse_args()

    color_map = {
        "corn": "blue",
        "rice": "orange",
        "cotton": "green",
        "soybean": "red",
        "winter_wheat": "purple",
    }

    df_values, df_labels = load_data(args.path_valores_personalizado, args.path_etiquetas_personalizado)
    #Escalar los datos menos la primera columna que es el id
    df_values_scl = df_values.iloc[:, 1:]
    df_values_scl = scale_data(df_values_scl)

    #Concatenar la primera columna que es el id con los datos escalados
    df_values = pd.concat([df_values.iloc[:, 0], df_values_scl], axis=1)

    if args.verbosidad:
        print("Header Valores")
        print(df_values.head())

        print("Header Etiquetas")
        print(df_labels.head())

        print("Shape Valores")
        print(df_values.shape)

        print("Shape Etiquetas")
        print(df_labels.shape)

    # Calcular el número de filas y columnas
    num_filas = (args.cantidad + 2) // 3  # Redondear hacia arriba para obtener el número de filas necesario
    num_columnas = 3

    # Crear una figura con subgráficos
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(15, 5 * num_filas))
    axs = axs.flatten()  # Aplanar la matriz de ejes para facilitar la iteración

    first_class = None

    count = 0
    iteracion = -1
    while count < args.cantidad and args.indice + iteracion < len(df_values):
        iteracion += 1
        ax = axs[count]
        id = df_values.iloc[args.indice + iteracion, 0]
        ejemplo = df_values.iloc[args.indice + count, 1:]  # excluimos la primera columna que es el id

        id_label = df_labels.iloc[args.indice + iteracion, 0]
        if id_label != id:
            raise ValueError("Error: El id del ejemplo no coincide con el id de la etiqueta")

        label = df_labels.iloc[args.indice + iteracion, 1]

        if first_class is None:
            first_class = label

        if args.same_class and label != first_class:
            continue

        eje_x = range(args.longitud_ejemplo)

        ax.set_title(f"Ejemplo {id} - Tipo Cultivo '{label}'")
        ax.plot(eje_x, ejemplo, color=color_map[label])
        

        # Establecer los límites de los ejes y para que todas las gráficas tengan la misma escala
        ax.set_ylim(0, 1)

        count += 1
    # Ocultar ejes no utilizados
    for j in range(args.cantidad, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if args.mostrar_ejemplo:
        plt.show()

    if args.guardar_ejemplo:
        plt.savefig("ejemplos.png")
