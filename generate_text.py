# Description: This script loads a pre-trained model and generates text samples.
# The model is trained on the Yelp dataset and is a character-level LSTM language model.
# The script generates samples of text by sampling from the model's distribution.
# The samples are printed to the console.
# The script requires the following files:
#     - config.py: contains the model parameters
#     - models/generator.py: contains the LSTMGenerator class
#     - models/SeqGAN_G.py: contains the SeqGAN_G class
#     - save/20240423/pp_tweets/maligan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl44_temp1_lfd0.0_T0423_1638_38/models/gen_MLE_00079.pt: the pre-trained model


import json
import torch
from models.generator import LSTMGenerator
from models.SeqGAN_G import SeqGAN_G
from models.LeakGAN_G import LeakGAN_G
from models.MaliGAN_G import MaliGAN_G
from models.JSDGAN_G import JSDGAN_G
from models.RelGAN_G import RelGAN_G
from models.DPGAN_G import DPGAN_G
from models.DGSAN_G import DGSAN_G
from models.CoT_G import CoT_G
from models.SentiGAN_G import SentiGAN_G
from models.CatGAN_G import CatGAN_G
from data_process.gen_dataset import sanitize_tweet
from utils.text_process import load_dict, tensor_to_tokens
from utils.data_loader import get_tokenlized_words
import os


def parse_log_file(log_file_path):
    """Parse and clean the log file to extract model parameters."""
    params = {}
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>>>'):
                # Remove leading '>>> ' and clean the line
                clean_line = line[4:].strip()
                key, value = clean_line.split(':', 1)
                key = key.strip()
                value = value.strip()
                params[key] = value
    return params

def load_generator(model_class, model_path, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,device, gpu=False):
    models_dict = {
        '' : LSTMGenerator,
        'seqgan': SeqGAN_G,
        'leakgan': LeakGAN_G,
        'maligan': MaliGAN_G,           
        'jsdgan': JSDGAN_G,
        'dpgan': DPGAN_G,
        'relgan': RelGAN_G,
        'sentigan': SentiGAN_G,
        'catgan': CatGAN_G,
        'dgsan': DGSAN_G,
        'cot': CoT_G
    }

    generator = models_dict[model_class](embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
    generator.load_state_dict(torch.load(model_path,map_location='cuda:{}'.format(device)))

    if gpu:
        generator = generator.cuda()
    generator.eval()
    return generator

def generate_tweets(generator, num_samples, batch_size, idx2word_dict, start_letter):
    tweets = ""
    try:
        with torch.no_grad():
            generator.init_hidden(batch_size)
            samples = generator.sample(num_samples, batch_size, start_letter)
            # samples = generator.sample_and_plot(num_samples, batch_size, start_letter)
            
            tokens = tensor_to_tokens(samples, idx2word_dict)
            
            for sent in tokens:
                    tweets+=(' '.join(sent))
                    tweets+=('\n\n')
    except Exception as e:
        raise Exception("Error generating tweets.", e)
    return tweets


def get_current_path():
    return os.path.dirname(os.path.abspath(__file__))

#OBLIGATIOR LLAMARME PORQUE SINO NO FUNCIONA
def CHANGE_CURRENT_DIR_MUST():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    

def list_available_politicians():
    save_folder = './save'
    politicians = []
    for date_folder in os.listdir(save_folder):
        date_folder_path = os.path.join(save_folder, date_folder)
        if os.path.isdir(date_folder_path):
            for party_folder in os.listdir(date_folder_path):
                party_folder_path = os.path.join(date_folder_path, party_folder)
                if os.path.isdir(party_folder_path):
                    politician = party_folder.split('_')[0]
                    if politician not in politicians:
                        politicians.append(politician)
    return politicians

def available_models_partido(politician):
    save_folder = './save'
    models_info = []
    for date_folder in os.listdir(save_folder):
        date_folder_path = os.path.join(save_folder, date_folder)
        if os.path.isdir(date_folder_path):
            for party_folder in os.listdir(date_folder_path):
                party_folder_path = os.path.join(date_folder_path, party_folder)
                if os.path.isdir(party_folder_path):
                    current_politician = party_folder.split('_')[0]
                    if current_politician == politician:
                        for model_folder in os.listdir(party_folder_path):
                            model_folder_path = os.path.join(party_folder_path, model_folder)
                            if os.path.isdir(model_folder_path):
                                models_subfolder_path = os.path.join(model_folder_path, 'models')
                                gen_models = []
                                dis_models = []
                                if os.path.isdir(models_subfolder_path):
                                    for model_file in os.listdir(models_subfolder_path):
                                        model_file_path = os.path.join(models_subfolder_path, model_file)
                                        if os.path.isfile(model_file_path):
                                            if 'gen' in model_file:
                                                gen_models.append(model_file)
                                            elif 'dis' in model_file:
                                                dis_models.append(model_file)
                                models_info.append({
                                    "model_name": model_folder,
                                    "model_path": model_folder_path,
                                    "gen_models": gen_models,
                                    "dis_models": dis_models
                                })
    return models_info

def get_all_data_json():
    partidos_politicos = list_available_politicians()
    data = {"partidos": []}
    
    for partido in partidos_politicos:
        modelos = available_models_partido(partido)
        data["partidos"].append({
            "nombre": partido,
            "modelos": modelos
        })
    
    return data

def main(load_model_path,gen_model, word = 'BOS'):
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.abspath(__file__)))
    # data_path = './save/20240711/pp_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl153_temp1_lfd0.0_T0711_1010_57/'
    # data_path = f'./save/20240711/{partido}_tweets/{model}/'
    
    log_file_path = os.path.join(load_model_path, 'log.txt')
    if not os.path.exists(log_file_path):
        raise FileNotFoundError("Log file not found.", get_current_path())
    model_path = os.path.join(load_model_path, 'models', gen_model)
    
    params = parse_log_file(log_file_path)
    
    # print(params)
    
    model_class = params['run_model'] 
    embedding_dim = int(params['gen_embed_dim'])
    hidden_dim = int(params['gen_hidden_dim'])
    vocab_size = int(params['vocab_size'])
    max_seq_len = int(params['max_seq_len'])
    padding_idx = int(params['padding_idx'])
    gpu = int(params['cuda'])
    device = int(params['device'])
    dataset = params['dataset']
    
    # load dictionary
    print("Loading dictionary... ", dataset)
    try:
        word2idx_dict, idx2word_dict = load_dict(dataset)
    except FileNotFoundError:
        print("Dictionary not found. Using default dictionary.")
        raise FileNotFoundError("Dictionary not found.")
    
    generator = load_generator(model_class, model_path, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,device ,gpu)
    
    num_samples = 1
    batch_size = 1  # Tweets en paralelo que se obtendran
    start_letter = [f'{word}']
    
    if word != 'BOS':
        start_letter = get_tokenlized_words(start_letter)
        
    # start_letter = [sanitize_tweet(t) for t in start_letter]
    
    print("Start letter:", start_letter)
    try:
        start_letter = [int(word2idx_dict.get(sl[0])) for sl in start_letter]
    except TypeError:
        print("Start letter not found in dictionary. Using default start letter.")
        start_letter = [1]
    print("Start letter token:", start_letter)
    
    tweets = generate_tweets(generator, num_samples, batch_size,idx2word_dict, start_letter[0])

    return tweets

if __name__ == "__main__":
    pol = list_available_politicians()
    print(pol)
    models = available_models_partido(pol[0])
    print(models)
    # tweets = main()
    
    data = get_all_data_json()
    print(json.dumps(data, indent=4))

    print(main('./save/20240814/psoe_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl60_temp1_lfd0.0_T0814_1540_59/', 'gen_ADV_training_00038.pt', 'pedro'))
