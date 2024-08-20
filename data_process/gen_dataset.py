import pandas as pd
import os
import preprocessor as p # clean tweets, erasing hastag and mentions
import unidecode
from tqdm import tqdm

from nltk.corpus import stopwords
import nltk
import re

'''
Este fichero crea el dataset procesado a partir de los datos en crudo
Se limpian los tweets y se guardan en carpetas por partido político
En cada carpeta se guarda un archivo CSV con los datos originales y otro con los datos limpios en formato texto
'''


def sanitize_tweet(tweet):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI)

    tweet = tweet.lower()
    
    tweet = unidecode.unidecode(tweet)
    tweet = p.tokenize(tweet)
    tweet = re.sub(r'([.,!?¿¡:;(){}\[\]"\'_\-])', r' \1 ', tweet)  # Añade espacios alrededor de signos de puntuación
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Elimina espacios adicionales

    return tweet

if __name__ == "__main__":
    # Leer el archivo CSV
    df = pd.read_csv('tweets_politica_kaggle.csv', sep='\t')
    # df = pd.read_csv('test.csv', sep='\t')

    new_folder_name = 'partidos_final'
    for i in tqdm(range(df.shape[0])):
        # Limpiamos el tweet con tweet-preprocessor y lo pasamos a minúsculas
        df.loc[i, 'clean_tweet'] = sanitize_tweet(df.loc[i, 'tweet'])

    # Crear una carpeta para cada partido y guardar los datos
    for partido in df['partido'].unique():
        # Crear la carpeta si no existe
        carpeta_partido = os.path.join(new_folder_name, partido)
        os.makedirs(carpeta_partido, exist_ok=True)
        
        # Filtrar los tweets del partido actual
        tweets_partido = df[df['partido'] == partido]
        
        # Guardar los tweets del partido en un archivo CSV
        tweets_partido[['partido', 'timestamp', 'tweet', 'clean_tweet']].to_csv(os.path.join(carpeta_partido, f'{partido}_tweets.csv'), index=False)

        # Guardar los tweets del partido en un archivo de texto
        with open(os.path.join(carpeta_partido, f'orig_{partido}_tweets.txt'), 'w') as f:
            for tweet in tweets_partido['clean_tweet']:
                f.write(tweet + '\n')
