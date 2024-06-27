import torch
import torch.nn.functional as F
import os
from models.generator import LSTMGenerator
from models.SeqGAN_G import SeqGAN_G
from utils.text_process import load_dict
from config import CNNDiscriminator, GRUDiscriminator

def load_discriminator(model_class, model_path, params, gpu=False):
    if model_class == CNNDiscriminator:
        model = model_class(
            embed_dim=int(params['embedding_dim']),
            vocab_size=int(params['vocab_size']),
            filter_sizes=list(map(int, params['filter_sizes'].split(','))),
            num_filters=list(map(int, params['num_filters'].split(','))),
            padding_idx=int(params['padding_idx']),
            gpu=gpu,
            dropout=float(params['dropout'])
        )
    elif model_class == GRUDiscriminator:
        model = model_class(
            embedding_dim=int(params['embedding_dim']),
            vocab_size=int(params['vocab_size']),
            hidden_dim=int(params['hidden_dim']),
            feature_dim=int(params['feature_dim']),
            max_seq_len=int(params['max_seq_len']),
            padding_idx=int(params['padding_idx']),
            gpu=gpu,
            dropout=float(params['dropout'])
        )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if gpu else 'cpu')))
    if gpu:
        model = model.cuda()
    model.eval()
    return model

def classify_tweets(discriminator, tweets, idx2word_dict, padding_idx, gpu=False):
    tokenized_tweets = []
    for tweet in tweets:
        tokenized_tweet = [idx2word_dict.get(word, padding_idx) for word in tweet.split()]
        tokenized_tweets.append(tokenized_tweet)
    
    max_len = max(len(tweet) for tweet in tokenized_tweets)
    padded_tweets = torch.zeros(len(tweets), max_len, dtype=torch.long) + padding_idx

    for i, tweet in enumerate(tokenized_tweets):
        padded_tweets[i, :len(tweet)] = torch.tensor(tweet)

    if gpu:
        padded_tweets = padded_tweets.cuda()

    with torch.no_grad():
        logits = discriminator(padded_tweets)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)

    return predictions.cpu().numpy()

if __name__ == "__main__":
    gpu = torch.cuda.is_available()

    data_path = 'save_borrar/20240617/vox_tweets/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl156_temp1_lfd0.0_T0617_1802_46'
    log_file_path = os.path.join(data_path, 'log.txt')
    discriminator_model_path = os.path.join(data_path, 'models', 'dis_MLE_00119.pt')
    
    # Parse log file to get model parameters
    params = parse_log_file(log_file_path)
    
    # Load the dictionary
    dataset = params['dataset']
    word2idx_dict, idx2word_dict = load_dict(dataset)
    
    # Load the discriminator model
    discriminator_class = CNNDiscriminator  # or GRUDiscriminator depending on your use case
    discriminator = load_discriminator(discriminator_class, discriminator_model_path, params, gpu)

    # Tweets to classify
    tweets = [
        "This is a sample tweet for testing.",
        "Another example tweet to classify."
    ]

    # Classify the tweets
    predictions = classify_tweets(discriminator, tweets, word2idx_dict, int(params['padding_idx']), gpu)
    
    # Print the predictions
    for i, pred in enumerate(predictions):
        print(f"Tweet {i+1}: {'Party A' if pred == 0 else 'Party B'}")
