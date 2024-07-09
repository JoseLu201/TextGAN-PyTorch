import pandas as pd
import os
import preprocessor as p # clean tweets, erasing hastag and mentions
import unidecode
from tqdm import tqdm

from nltk.corpus import stopwords
import nltk
import re
import os

# def load_ADV_train_state(model_path):
#     files = [f for f in os.listdir(model_path) if 'ADV' in f]
#     if not files:
#         return None
#     latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#     return {
#         'gen' : os.path.join(model_path, latest_file),
#         'dis' : os.path.join(model_path, latest_file)
#         ''
#     }

def path_ADV_checkpoints(model_dir):
    files = [f for f in os.listdir(model_dir) if 'ADV' in f]
    if not files:
        return None

    # Find the highest epoch
    max_epoch = max(int(f.split('_')[-1].split('.')[0]) for f in files)
    paths = {}

    for f in files:
        epoch = int(f.split('_')[-1].split('.')[0])
        if epoch == max_epoch:
            model_type = f.split('_')[0]
            if model_type in ['gen', 'dis', 'clas']:
                paths[model_type] = os.path.join(model_dir, f)
    return paths

# print(path_ADV_checkpoints('save/20240701/ciudadanos_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl174_temp1_lfd0.0_T0701_0639_50/models'))

print(path_ADV_checkpoints('save/20240701/ciudadanos_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl174_temp1_lfd0.0_T0701_0639_50/models' ))