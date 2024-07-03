import pandas as pd
import os
import preprocessor as p # clean tweets, erasing hastag and mentions
import unidecode
from tqdm import tqdm

from nltk.corpus import stopwords
import nltk
import re


def sanitize_tweet(tweet):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI)

    tweet = tweet.lower()
    
    tweet = unidecode.unidecode(tweet)
    # tweet = ''.join(e if e.isalnum() or e.isspace() or e in ['.', ',', '!', '?', '¿', '¡', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '-', '_', '$'] else ' ' for e in tweet )
    tweet = p.tokenize(tweet)
    tweet = re.sub(r'([.,!?¿¡:;(){}\[\]"\'_\-])', r' \1 ', tweet)  # Añade espacios alrededor de signos de puntuación
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Elimina espacios adicionales

    return tweet



print(sanitize_tweet("hola.adios  . ! como? estsa #hola @adios como?"))