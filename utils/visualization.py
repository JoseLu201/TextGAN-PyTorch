# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : visualization.py
# @Time         : Created at 2019-03-19
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import matplotlib.pyplot as plt
import re

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
    'BLEU': 'BLEU',
    'Self-BLEU': 'Self-BLEU',
}

color_list = ['#e74c3c', '#e67e22', '#f1c40f', '#8e44ad', '#2980b9', '#27ae60', '#16a085']


def plt_data(data, step, title, c_id, savefig=False):
    x = [i for i in range(step)]
    
    x_label = 'Epoch'
    y_label = 'loss'
    plt.plot(x, data, label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if savefig:
        plt.savefig('savefig/' + title + '.png')


def get_log_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')
        data_dict = {'pre_loss': [], 'g_loss': [], 'mana_loss': [], 'work_loss': [],
                     'd_loss': [], 'train_acc': [], 'eval_acc': [], 'NLL_oracle': [],
                     'NLL_gen': [], 'BLEU': [], 'Self-BLEU': [],}

        for line in all_lines:
            items = line.split()
            # print(items)
            # input()
            try:
                for key in data_dict.keys():
                    if key == 'BLEU' or key == 'Self-BLEU':
                        bleu_found, array_data = process_log_line(line, regex_key=key)
                        if bleu_found:
                            data_dict[key].append(array_data)
                    elif key in items:
                        data_dict[key].append(float(items[items.index(key) + 2][:-1]))
                    
            except:
                break

    return data_dict

def process_log_line(line, regex_key='BLEU'):
    bleu_found = False
    array_data = None
    pattern = rf"{re.escape(regex_key)}-\[\d+(?:,\s*\d+)*\] = \[(.*?)\]"
    match = re.search(pattern, line)
    if match:
        bleu_found = True
        array_str = match.group(1)  # Captura el grupo que contiene el 
        array_data = [float(x) for x in array_str[1:-1].split(',')]  # Convierte la cadena en una lista de floats
    return bleu_found, array_data


if __name__ == '__main__':
    # log_file_root = '../log/'
    log_file_root = '../'
    # Custom your log files in lists, no more than len(color_list)
    
    # log_file_list = ['dpgan_log'] 
    log_file_list = ['mali_log']
    
    
    # legend_text = ['SeqGAN', 'LeakGAN', 'RelGAN']

    color_id = 0
    if_save = True
    # legend_text = log_file_list
    log_file = log_file_root + log_file_list[0] + '.txt'
    all_data = get_log_data(log_file)
    print(all_data)
    
    for idx, item in enumerate(title_dict):
        print(item + " " + title_dict[item])
        
        assert item in title_dict.keys(), 'Error data name'
        plt.clf()
        plt.title(item)
        
        if all_data[title_dict[item]] != []:
            plt_data(all_data[title_dict[item]], len(all_data[title_dict[item]]),
                    "Mali_"+item, color_id, if_save)
        # color_id += 1

        plt.legend()
        plt.show()
        # plt.savefig(f'./savefig/{item}.pdf')
