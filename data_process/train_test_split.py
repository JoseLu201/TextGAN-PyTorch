from sklearn.model_selection import train_test_split
import os
import shutil

'''
Este script divide los datos en conjuntos de entrenamiento y prueba
Entrada:
    - Nombre del partido político
    - Directorio donde se volcaran los datos

Se modicicara las carpetas creadas en gen_dataset.py añadiendo una carpeta 'test' con los datos de prueba y un archivo por 
ejemplo pp_tweets_test.txt con los datos de train.

Una vez separados los datos, se copiarán en la carpeta dataset para que puedan ser utilizados por el modelo.
'''

# Nombre del partido político
partido = "psoe"
dir = "partidos_final"
PATH = f"{dir}/{partido}/"
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
    os.makedirs(os.path.join(PATH, 'test'), exist_ok=True)
    percent = 1
    # os.makedirs(f"{PATH}/test")
    # Guardar los datos de entrenamiento en un archivo
    with open(os.path.join(PATH, train_file), 'w') as file:
        # file.writelines(X_train)
        file.writelines(X_train[:len(X_train)//percent])
        
    shutil.copy(os.path.join(PATH, train_file), os.path.join('../dataset', train_file))
    # Guardar los datos de prueba en un archivo
    with open(os.path.join(PATH, test_file), 'w') as file:
        # file.writelines(X_test)
        file.writelines(X_test[:len(X_train)//percent])
        
    shutil.copy(os.path.join(PATH, test_file), os.path.join('../dataset/testdata', test_file.split('/')[1]))
except Exception as e:
    print(e)