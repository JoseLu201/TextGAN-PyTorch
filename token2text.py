
import torch
from utils import text_process as tp

def load_tensor(filename):
  """
  Carga tokens desde un archivo.

  Args:
    filename: Ruta al archivo que contiene los tokens.

  Returns:
    Lista de tokens (palabras o sub-palabras).
  """

  tokens = ""
  with open(filename, 'r') as f:
    for line in f:
      tokens += str(line.strip().split())

  return tokens

word2idx_dict, idx2word_dict = tp.load_dict("pp_tweets")

model = torch.load('save/20240423/pp_tweets/maligan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl44_temp1_lfd0.0_T0423_1638_38/models/gen_MLE_00079.pt')
import torch
import config as cfg
# Specify the model parameters
embedding_dim = 32
hidden_dim = 32
vocab_size = 21096
max_seq_len = 20
padding_idx = 0
gpu = True  # Set to True if using GPU
from models.generator import LSTMGenerator
# Load the pre-trained model (replace 'checkpoint.pt' with your actual file path)
model = LSTMGenerator(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
model.load_state_dict(torch.load('save/20240423/pp_tweets/maligan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl44_temp1_lfd0.0_T0423_1638_38/models/gen_MLE_00079.pt'))
model.eval()
if gpu:
    model = model.cuda()

# Generate a sample of text
num_samples = 10  # Generate 1 sample
sample = model.sample(num_samples,64, start_letter=192)

# Decode the sample from word indices to text
if gpu:
    sample = sample.cpu()
sample = sample.data.numpy()



# Print the generated text
for i in range(sample.shape[0]):
  # print(f"\'{sample[i][0]}\'")
  # print(idx2word_dict.get(f"{sample[i][0]}"))
  seq = [idx2word_dict.get(f'{idx}') for idx in sample[i]]
  print(' '.join(seq))


  
