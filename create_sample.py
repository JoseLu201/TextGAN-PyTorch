import torch
from instructor.real_data.seqgan_instructor import SeqGAN_G  # Import only SeqGAN_G
from utils.text_process import load_dict, tensor_to_tokens
import config as cfg
from instructor.real_data.instructor import BasicInstructor



# ===Program===
if_test = int(False)
run_model = 'seqgan'
CUDA = int(True)
oracle_pretrain = int(True)
gen_pretrain = int(False)
dis_pretrain = int(False)
MLE_train_epoch = 120
ADV_train_epoch = 200
tips = 'SeqGAN experiments'

# ===Oracle  or Real===
if_real_data = [int(False), int(True), int(True)]
dataset = ['oracle', 'image_coco', 'emnlp_news']
vocab_size = [5000, 0, 0]

# ===Basic Param===
data_shuffle = int(False)
model_type = 'vanilla'
gen_init = 'normal'
dis_init = 'uniform'
samples_num = 10000
batch_size = 64
max_seq_len = 20
gen_lr = 0.01
dis_lr = 1e-4
pre_log_step = 10
adv_log_step = 1

# ===Generator===
ADV_g_step = 1
rollout_num = 16
gen_embed_dim = 32
gen_hidden_dim = 32

# ===Discriminator===
d_step = 5
d_epoch = 3
ADV_d_step = 4
ADV_d_epoch = 2
dis_embed_dim = 64
dis_hidden_dim = 64

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(True)
use_nll_div = int(True)
use_bleu = int(True)
use_self_bleu = int(True)
use_ppl = int(False)
job_id = 2
args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--cuda', CUDA,
    # '--device', gpu_id,  # comment for auto GPU
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips,

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--vocab_size', vocab_size[job_id],

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--samples_num', samples_num,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--rollout_num', rollout_num,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,

    # Discriminator
    '--d_step', d_step,
    '--d_epoch', d_epoch,
    '--adv_d_step', ADV_d_step,
    '--adv_d_epoch', ADV_d_epoch,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_ppl', use_ppl,
]

args = dict(map(str, args))


ruta_modelo_entrenado = 'save/20240330/emnlp_news/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl51_temp1_lfd0.0_T0330_1038_55/models/gen_MLE_00119.pt'
word2idx_dict, idx2word_dict = load_dict('emnlp_news')
num_samples = 5  # Number of samples to generate
max_len = 50  # Maximum sequence length

# Create generator model
model = BasicInstructor(args)
model.init_model()
model._save('ADV',0)
# Load pre-trained model parameters
model.load_state_dict(torch.load(ruta_modelo_entrenado))
model.eval()  # Set to evaluation mode

# Generate samples
samples = model.sample(num_samples, max_len)

# Convert and print text
generated_text = [tensor_to_tokens(sample, idx2word_dict) for sample in samples]
for text in generated_text:
  print(" ".join(text))
