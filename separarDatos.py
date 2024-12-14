from sklearn.model_selection import train_test_split
import pandas as pd

LONGITUD_EJEMPLO = 131
PATH_VALORES = 'X_train.csv'
PATH_ETIQUETAS = 'Y_train.csv'


df_valores = pd.read_csv(PATH_VALORES, delimiter=';')
df_etiquetas = pd.read_csv(PATH_ETIQUETAS, delimiter=';')

# Supongamos que X e Y son tus datos de entrada y etiquetas respectivamente
X_train, X_test, Y_train, Y_test = train_test_split(df_valores, df_etiquetas, test_size=0.10, random_state=1234, stratify=df_etiquetas.iloc[:, 1])


# Visualizar la cantidad y porcentaje de cada clase en train
print("Cantidad de cada clase en train:")
for clase, cantidad in Y_train.iloc[:, 1].value_counts().items():
    porcentaje = Y_train.iloc[:, 1].value_counts(normalize=True)[clase] * 100
    print(f"{clase:<15} {cantidad} ({porcentaje:.2f}%)")

# Visualizar la cantidad y porcentaje de cada clase en test
print("Cantidad de cada clase en test:")
for clase, cantidad in Y_test.iloc[:, 1].value_counts().items():
    porcentaje = Y_test.iloc[:, 1].value_counts(normalize=True)[clase] * 100
    print(f"{clase:<15} {cantidad} ({porcentaje:.2f}%)")


OUTPUT_PATH = "datos_separados"

X_train.to_csv(f"{OUTPUT_PATH}/X_train.csv", index=False, sep=';')
Y_train.to_csv(f"{OUTPUT_PATH}/Y_train.csv", index=False, sep=';')
X_test.to_csv(f"{OUTPUT_PATH}/X_test.csv", index=False, sep=';')
Y_test.to_csv(f"{OUTPUT_PATH}/Y_test.csv", index=False, sep=';')
