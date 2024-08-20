# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : visualization.py
# @Time         : Created at 2019-03-19
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import matplotlib.pyplot as plt

title_dict = {
    'gen_pre_loss': 'pre_loss',
    'gen_adv_loss': 'g_loss',
    'gen_mana_loss': 'mana_loss',
    'gen_work_loss': 'work_loss',
    'dis_loss': 'd_loss',
    'dis_train_acc': 'train_acc',
    'dis_eval_acc': 'eval_acc',
    'NLL_oracle': 'NLL_oracle',
    'NLL_gen': 'NLL_gen',
    'BLEU-3': 'BLEU-3',
}

color_list = ['#e74c3c', '#e67e22', '#f1c40f', '#8e44ad', '#2980b9', '#27ae60', '#16a085']


def plt_data(data, step, title, c_id, savefig=False):
    x = [i for i in range(step)]
    plt.plot(x, data, color=color_list[c_id], label=title)
    if savefig:
        plt.savefig('savefig/' + title + '.png')


def get_log_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')
        data_dict = {'pre_loss': [], 'g_loss': [], 'mana_loss': [], 'work_loss': [],
                     'd_loss': [], 'train_acc': [], 'eval_acc': [], 'NLL_oracle': [],
                     'NLL_gen': [], 'BLEU-3': []}

        for line in all_lines:
            items = line.split()
            try:
                for key in data_dict.keys():
                    if key in items:
                        data_dict[key].append(float(items[items.index(key) + 2][:-1]))
            except:
                break

    return data_dict


if __name__ == '__main__':
    # log_file_root = '../log/'
    log_file_root = '../'
    
    # Custom your log files in lists, no more than len(color_list)
    log_file_list = ['logtest']
    legend_text = ['gen_pre_loss','gen_mana_loss','gen_work_loss']

    color_id = 0
    data_name = 'gen_adv_loss'
    if_save = False
    # legend_text = log_file_list

    assert data_name in title_dict.keys(), 'Error data name'
    all_data_list = []
    log_file = log_file_root + log_file_list[0]+'.txt'
    all_data = get_log_data(log_file)
    for idx, key in enumerate(title_dict.keys()):
        if key.startswith('gen') and key != "gen_adv_loss":
            data_name = key
            print(data_name)
            plt.title(data_name)
            plt_data(all_data[title_dict[data_name]], len(all_data[title_dict[data_name]]),
                    legend_text[idx], color_id, if_save)
            color_id += 1

    plt.legend()
    plt.show()
    plt.savefig('../saved.pdf')