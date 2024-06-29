import torch
import torch.nn.functional as F
import os
from utils.text_process import load_dict
from data_process.gen_dataset import sanitize_tweet
from utils.data_loader import GenDataIter

from models.discriminator import CNNClassifier, GRUDiscriminator, CNNDiscriminator
from models.SeqGAN_D import SeqGAN_D
from models.LeakGAN_D import LeakGAN_D
from models.MaliGAN_D import MaliGAN_D
# from models.JSDGAN_G import JSDGAN_D
from models.RelGAN_D import RelGAN_D
from models.DPGAN_D import DPGAN_D
# from models.DGSAN_D import DGSAN_D
from models.CoT_D import Cot_D
from models.SentiGAN_D import SentiGAN_D
from models.CatGAN_D import CatGAN_D
import config as cfg
import io
from utils.data_loader import get_tokenlized

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

def load_discriminator(model_class, model_path, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False, dropout=0.2):
    models_dict = {
        'seqgan': SeqGAN_D,
        'leakgan': LeakGAN_D,
        'maligan': MaliGAN_D,
        # 'jsdgan': JSDGAN_D,
        'dpgan': DPGAN_D,
        'relgan': RelGAN_D,
        'sentigan': SentiGAN_D,
        'catgan': CatGAN_D,
        # 'dgsan': DGSAN_D,
        'cot': Cot_D
    }

    
    dis = models_dict[model_class](embed_dim=embedding_dim, vocab_size=vocab_size, padding_idx=padding_idx,gpu=gpu, dropout=dropout)
    dis.load_state_dict(torch.load(model_path))
    if gpu:
        dis = dis.cuda()
    dis.eval()
    return dis

def classify_tweets(discriminator, file_tweets, word2idx_dict, padding_idx, gpu=False):
    
    data_loader = GenDataIter(file_tweets).loader
    print(f"Data Loader Length: {type(data_loader.dataset.data)}")
    print(f"Data Loader Length: {len(data_loader.dataset.data)}")
    
    for data in (data_loader.dataset):
        print("INpt")
        inp, target = data['input'], data['target']
        inp = inp.unsqueeze(0)

        # Print the shape of inp to debug
        print("Shape of inp:", inp.shape)
        
        if cfg.CUDA:
            inp, target = inp.cuda(), target.cuda()

        pred = discriminator.forward(inp)   
        print("Pred", pred)
        with torch.no_grad():
            logits = discriminator.forward(inp)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

        return predictions.cpu().numpy()
    

if __name__ == "__main__":
    gpu = torch.cuda.is_available()

    data_path = 'save_borrar/20240617/vox_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl156_temp1_lfd0.0_T0617_1802_46'
    log_file_path = os.path.join(data_path, 'log.txt')
    model_path = os.path.join(data_path, 'models', 'dis_MLE_00119.pt')
    
    # Parse log file to get model parameters
    params = parse_log_file(log_file_path)
    
    # print(params)
    
    model_class = params['run_model'] 
    embedding_dim = int(params['dis_embed_dim'])
    hidden_dim = int(params['dis_hidden_dim'])
    vocab_size = int(params['vocab_size'])
    max_seq_len = int(params['max_seq_len'])
    padding_idx = int(params['padding_idx'])
    gpu = int(params['cuda'])
    dataset = params['dataset']
    cfg.dataset = dataset
    
    cfg.batch_size =1# int(params['batch_size'])
    cfg.max_seq_len = max_seq_len
    cfg.start_letter = int(params['start_letter'])
    cfg.data_shuffle = bool(params['shuffle'])
    cfg.if_real_data = bool(params['if_real_data'])

    
    # Load the dictionary
    word2idx_dict, idx2word_dict = load_dict(dataset)
    
    discriminator = load_discriminator(model_class, model_path, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)

    # Tweets to classify
    tweets = [
        "todo el psoe aplaudiendo a franco, fundador de la seguridad social.son tan ignorantes que, pese a la mala fe, a veces aciertan. https://t.co/1zvzvzvzvzv",
        "no hay que olvidar que el psoe es el partido de los ERES",
        "el psoe es el partido de los ERES",
        "todo el psoe aplaudiendo a franco, fundador de la seguridad social.son tan ignorantes que, pese a la mala fe, a veces aciertan. https://t.co/1zvzvzvzvzv",

    ]
    
    tweets_sani = [sanitize_tweet(t) for t in tweets]
    print(tweets_sani)
  

    tmp_tweets = "dis_sani_tweets.txt"

    with open(tmp_tweets, 'w') as f:
        for idx, tweet in enumerate(tweets_sani):
            f.write(tweet + '\n')
    
    # Classify the tweets
    predictions = classify_tweets(discriminator, tmp_tweets, word2idx_dict, padding_idx, gpu)
    print("Predictions:", predictions)
