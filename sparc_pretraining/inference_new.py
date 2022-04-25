import os
import sys
import time
import json
import torch
import random
import argparse
import operator
import progressbar
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
from tqdm import tqdm
import json

def load_data(data_path):
    print('Loading data from {}'.format(data_path))
    with open(data_path) as f:
        data = json.load(f)

    return data



def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--dataset_prefix_path', type=str,
                        default='../data/text2sql/data/sparc_tokenized_uncased_with_rc_tw_with_final/',
                        help='The path where the data stores.')

    # pretraining datasets specification
    parser.add_argument('--use_nlu', type=str, default='False', help='whether using NLU data during pretraining.')
    parser.add_argument('--use_bs', type=str, default='False', help='whether using DST data during pretraining.')
    parser.add_argument('--use_da', type=str, default='False', help='whether using POL data during pretraining.')
    parser.add_argument('--use_nlg', type=str, default='False', help='whether using NLG data during pretraining.')
    parser.add_argument('--use_sqlg', type=str, default='True', help='whether using NL2SQL data during pretraining.')
    parser.add_argument('--use_rcg', type=str, default='False', help='whether using NL2SQL data during pretraining.')
    parser.add_argument('--use_twg', type=str, default='False', help='whether using NL2TW data during pretraining.')
    parser.add_argument('--use_fug', type=str, default='False', help='whether using FUG data during pretraining.')


    parser.add_argument('--pretrained_path', type=str, default='t5_base_baseline_uncased/base_with_final_rc_tw/',
                        help='the path that stores pretrained checkpoint.')

    return parser.parse_args()


import argparse

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        device = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print('Using single GPU training.')
    else:
        device = torch.device('cpu')

    args = parse_config()


    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)

    from modelling.T5Model import T5Gen_Model

    special_token_list = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>',
                          '<f_key>', '<sos_context>', '<eos_context>', '<table>', '<number>', '<text>', '<boolean>',
                          '<others>', '<time>', '<column>', '<sos_rc>', '<eos_rc>', '<sos_tw>', '<eos_tw>']

    T5_model = T5Gen_Model(args.pretrained_path, tokenizer, special_token_list, dropout=0.0,
                            add_special_decoder_token=True, is_training=False)

    if cuda_available:
        T5_model = T5_model.to(device)

    T5_model.eval()
    print('Model loaded')

    sos_context_token_id = tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
    eos_context_token_id = tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]

    eos_q_token_id = tokenizer.convert_tokens_to_ids(['<eos_q>'])[0]
    sos_q_token_id = tokenizer.convert_tokens_to_ids(['<sos_q>'])[0]
    pad_token_id = tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
    start_token, end_token = '<sos_q>', '<eos_q>'

    # construct task-specific prefix
    sqlg_prefix_text = 'translate dialogue to system query:'
    sqlg_predix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sqlg_prefix_text))

    f = open('sparc_predict/test.txt', 'w')
    with torch.no_grad():
        dev_data = load_data(args.dataset_prefix_path + 'dev.json')
        for one_sess in tqdm(dev_data):
            schema_id_list = one_sess['schema_id_list']  # this is the schema ids
            dial_sess_list = one_sess["dialogue_session"]  # this list contains all turns from on session

            previous_context = []

            turn_num = len(dial_sess_list) - 1

            for turn_id in range(turn_num):
                curr_turn = dial_sess_list[turn_id]
                curr_turn_list = []
                curr_user_input = curr_turn['user_id_list']

                # ----------------------------------------------------------- #
                sqlg_input = previous_context + curr_user_input
                sqlg_input = sqlg_input + schema_id_list
                sqlg_input = sqlg_predix_id + [sos_context_token_id] + sqlg_input[-900:] + \
                                 [eos_context_token_id]
                sqlg_input = torch.LongTensor([sqlg_input]).to(device)

                outputs = T5_model.model.generate(input_ids=sqlg_input, decoder_start_token_id=sos_q_token_id,
                                              pad_token_id=pad_token_id, eos_token_id=eos_q_token_id,
                                              max_length=128)
                one_res_text = T5_model.tokenized_decode(outputs[0])
                one_res_text = one_res_text.strip()

                final_res_list = []
                for token in one_res_text.split():
                    if token == '<_PAD_>':
                        continue
                    else:
                        final_res_list.append(token)
                one_res_text = ' '.join(final_res_list).strip()


                curr_sys_query = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_res_text))
                previous_context = previous_context + curr_user_input + curr_sys_query

                final = one_res_text.replace(start_token, '').replace(end_token, '').strip()
                final = final.replace(' . ', '.').replace(' , ', ', ').replace('<unk>', '<')
                f.write(final + '\n')
            f.write('\n')
    f.close()






