from sklearn.model_selection import train_test_split
import os

# Nombre del partido político
partido = "pp"

PATH = f"partidos/{partido}/"
FILE = f"orig_{partido}_tweets.txt"

# Leer los datos del archivo
with open(os.path.join(PATH, FILE), 'r') as file:
    # datos = file.readlines()
    datos = [linea for linea in file if linea.strip()]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test = train_test_split(datos, test_size=0.2, random_state=42)

# Definir los nombres de los archivos de salida
train_file = f"{partido}_tweets.txt"
test_file = f"test/{partido}_tweets_test.txt"

try:
    # os.makedirs(f"{PATH}/test")
    # Guardar los datos de entrenamiento en un archivo
    with open(os.path.join(PATH, train_file), 'w') as file:
        file.writelines(X_train)

    # Guardar los datos de prueba en un archivo
    with open(os.path.join(PATH, test_file), 'w') as file:
        file.writelines(X_test)

except Exception as e:
    print(e)