# Description: This script loads a pre-trained model and generates text samples.
# The model is trained on the Yelp dataset and is a character-level LSTM language model.
# The script generates samples of text by sampling from the model's distribution.
# The samples are printed to the console.
# The script requires the following files:
#     - config.py: contains the model parameters
#     - models/generator.py: contains the LSTMGenerator class
#     - models/SeqGAN_G.py: contains the SeqGAN_G class
#     - save/20240423/pp_tweets/maligan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl44_temp1_lfd0.0_T0423_1638_38/models/gen_MLE_00079.pt: the pre-trained model



import torch
import config as cfg
from models.generator import LSTMGenerator
from utils.text_process import load_dict
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

def load_generator(model_path, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
    generator = LSTMGenerator(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
    generator.load_state_dict(torch.load(model_path))
    if gpu:
        generator = generator.cuda()
    generator.eval()
    return generator

def generate_tweets(generator, num_samples, batch_size, start_letter, gpu=False):
    samples = generator.sample(num_samples, batch_size, start_letter)
    return samples

def idx_to_word(samples, idx2word_dict, padding_idx):
    texts = []
    for sample in samples:
        text = ' '.join([idx2word_dict.get(f'{idx.item()}', '1') for idx in sample if idx.item() != padding_idx])
        texts.append(text)
    return texts


if __name__ == "__main__":
    
    
    data_path = 'save_borrar/20240617/vox_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl156_temp1_lfd0.0_T0617_1802_46'
    log_file_path = os.path.join(data_path, 'log.txt')
    model_path = os.path.join(data_path, 'models', 'gen_MLE_00119.pt')
    
    params = parse_log_file(log_file_path)
    
    print(params)
    
    embedding_dim = int(params['gen_embed_dim'])
    hidden_dim = int(params['gen_hidden_dim'])
    vocab_size = int(params['vocab_size'])
    max_seq_len = int(params['max_seq_len'])
    padding_idx = int(params['padding_idx'])
    gpu = int(params['cuda'])
    dataset = params['dataset']

    generator = load_generator(model_path, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
    
    num_samples = 2
    batch_size = 2  # Tweets en paralelo que se obtendran
    start_letter = cfg.start_letter
    samples = generate_tweets(generator, num_samples, batch_size, start_letter, gpu)
    
    # load dictionary
    word2idx_dict, idx2word_dict = load_dict(dataset)
    
    tweets = idx_to_word(samples, idx2word_dict, padding_idx) 

    for i, tweet in enumerate(tweets):
        print(f"Tweet {i+1}: {tweet}")