
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

# Example tokenized sequence
# tokens = load_tokens('save/20240329/psoe_tweets/maligan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl20_temp1_lfd0.0_T0329_1639_16/samples/samples_ADV_00054.txt')
tensor = tp.read_tensor('save/20240329/pp_tweets/cot_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl20_temp1_lfd0.0_T0329_2241_24/samples/samples_ADV_19999.txt')
tensor = torch.tensor(tensor)
# print(tensor)

# Initialize empty sentence string
data = tp.tensor_to_tokens(tensor,idx2word_dict )
for d in  data:
  print(f">>>: {d}")

  
