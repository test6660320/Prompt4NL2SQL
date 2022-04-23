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


def e2e_batch_generate(model, one_inference_batch, input_contain_db, data):
    is_cuda = next(model.parameters()).is_cuda
    if is_cuda:
        #device = next(model.parameters()).device
        device = torch.device('cuda')
        if torch.cuda.device_count() > 1: # multi-gpu training
            model = model.module
        else: # single gpu training
            pass
    else:
        device = 0

    max_span_len, max_response_len = 80, 120
    tokenizer = data.tokenizer
    bs_batch, da_batch, nlg_batch, parse_dict_batch = one_inference_batch
    batch_size = len(parse_dict_batch)
    res_batch_parse_dict = parse_dict_batch

    # if input_contain_db == True: then we first generate the belief state and get the db result
    # otherwise, we perform the generation all in-parallel

    bs_tensor, bs_mask = data.pad_batch(bs_batch)
    if is_cuda:
        bs_tensor = bs_tensor.cuda(device)
        bs_mask = bs_mask.cuda(device)

    batch_bs_text = model.batch_generate(bs_tensor, bs_mask, generate_mode='bs', max_decode_len=max_response_len)

    # the belief state sequence could be long
    batch_bs_restore_text = []
    for idx in range(batch_size):
        one_bs_text = batch_bs_text[idx]
        res_batch_parse_dict[idx]['bspn_gen'] = one_bs_text

    if input_contain_db:
        # we need to query the db base
        batch_db_input_id_list = []
        for idx in range(batch_size):
            one_queried_db_result = \
            data.reader.bspan_to_DBpointer(batch_bs_text[idx], res_batch_parse_dict[idx]['turn_domain'])
            one_db_text = '<sos_db> ' + one_queried_db_result + ' <eos_db>'
            one_db_token_id_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_db_text))
            batch_db_input_id_list.append(one_db_token_id_input)
    else:
        batch_db_input_id_list = []
        for _ in range(batch_size):
            batch_db_input_id_list.append([])

    if input_contain_db:
        # then we generate the dialogue action
        da_batch_input_id_list = []
        for idx in range(batch_size):
            one_da_input_id_list = da_batch[idx] + batch_db_input_id_list[idx]
            da_batch_input_id_list.append(one_da_input_id_list)
        da_tensor, da_mask = data.pad_batch(da_batch_input_id_list)
    else:
        da_tensor, da_mask = data.pad_batch(da_batch)

    if is_cuda:
        da_tensor = da_tensor.cuda(device)
        da_mask = da_mask.cuda(device)
    batch_da_text = model.batch_generate(da_tensor, da_mask, generate_mode='da', max_decode_len=max_span_len)


    for idx in range(batch_size):
        res_batch_parse_dict[idx]['aspn_gen'] = batch_da_text[idx]

    if input_contain_db:
        # finally, we generate the response
        nlg_batch_input_id_list = []
        for idx in range(batch_size):
            one_nlg_input_id_list = nlg_batch[idx] + batch_db_input_id_list[idx]
            nlg_batch_input_id_list.append(one_nlg_input_id_list)
        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch_input_id_list)
    else:
        nlg_tensor, nlg_mask = data.pad_batch(nlg_batch)

    if is_cuda:
        nlg_tensor = nlg_tensor.cuda(device)
        nlg_mask = nlg_mask.cuda(device)
    batch_nlg_text = model.batch_generate(nlg_tensor, nlg_mask, generate_mode='nlg', max_decode_len=max_response_len)
    for idx in range(batch_size):
        res_batch_parse_dict[idx]['resp_gen'] = batch_nlg_text[idx]
    return res_batch_parse_dict


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--dataset_prefix_path', type=str, default='../data/text2sql/data/sparc_tokenized_uncased_with_rc_tw_with_final/', help='The path where the data stores.')

    parser.add_argument('--shuffle_mode', type=str, default='session_level',
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    # pretraining datasets specification
    parser.add_argument('--use_nlu', type=str, default='False', help='whether using NLU data during pretraining.')
    parser.add_argument('--use_bs', type=str, default='False', help='whether using DST data during pretraining.')
    parser.add_argument('--use_da', type=str, default='False', help='whether using POL data during pretraining.')
    parser.add_argument('--use_nlg', type=str, default='False', help='whether using NLG data during pretraining.')
    parser.add_argument('--use_sqlg', type=str, default='True', help='whether using NL2SQL data during pretraining.')
    parser.add_argument('--use_rcg', type=str, default='False', help='whether using NL2SQL data during pretraining.')
    parser.add_argument('--use_twg', type=str, default='False', help='whether using NL2TW data during pretraining.')
    parser.add_argument('--use_fug', type=str, default='False', help='whether using FUG data during pretraining.')

    parser.add_argument('--add_prefix', type=str, default='True',
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    # model configuration
    parser.add_argument('--model_name', type=str, default='t5-large', help='t5-small or t5-base or t5-large')

    parser.add_argument('--pretrained_path', type=str, default='t5_base_baseline_uncased/base_with_final_rc_tw_4e-5/', help='the path that stores pretrained checkpoint.')
    # training configuration
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")
    parser.add_argument("--output_save_path", type=str, help="directory to save the model output.")
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
 
    args = parse_config()
    device = torch.device('cuda')

    print('Start loading data...')
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)

    if args.add_prefix == 'True':
        add_prefix = True
    elif args.add_prefix == 'False':
        add_prefix = False
    else:
        raise Exception('Wrong Prefix Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    from dataclass import TOD_PRETRAINING_CORPUS
    # from dataclass_simplified import TOD_PRETRAINING_CORPUS

    data = TOD_PRETRAINING_CORPUS(tokenizer, args.shuffle_mode, args.dataset_prefix_path, use_nlu=args.use_nlu,
                                  use_bs=args.use_bs, use_da=args.use_da, use_nlg=args.use_nlg, use_sqlg=args.use_sqlg,
                                  use_rcg=args.use_rcg, use_twg=args.use_twg, use_fug=args.use_fug, max_tgt_len=128)
    print ('Data Loaded.')

    print('Start loading model...')
    assert args.model_name.startswith('t5')
    from modelling.T5Model import T5Gen_Model
    # special_token_list = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>', '<f_key>',
    #                   '<sos_context>', '<eos_context>','<table>', '<number>', '<text>', '<boolean>', '<others>', '<time>']
    # special_token_list = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>', '<f_key>',
    #                   '<sos_context>', '<eos_context>', '<table>', '<number>', '<text>', '<boolean>', '<others>',
    #                   '<time>', '<column>', '<sos_rc>', '<eos_rc>']
    special_token_list = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>',
                          '<f_key>', '<sos_context>', '<eos_context>', '<table>', '<number>', '<text>', '<boolean>',
                          '<others>', '<time>', '<column>', '<sos_rc>', '<eos_rc>', '<sos_tw>', '<eos_tw>']
    # special_token_list = ['<_PAD_>', '<sos_u>', '<eos_u>', '<sos_q>', '<eos_q>', '<sos_s>', '<eos_s>', '<p_key>', '<f_key>',
    #                   '<sos_context>', '<eos_context>', '<table>', '<number>', '<text>', '<boolean>', '<others>',
    #                   '<time>', '<column>', '<sos_rc>', '<eos_rc>', '<sos_tw>', '<eos_tw>', '<sos_gf>', '<eos_gf>']

    if args.pretrained_path != 'None':
        model = T5Gen_Model(args.pretrained_path, data.tokenizer, special_token_list, dropout=0.0,
                            add_special_decoder_token=add_special_decoder_token, is_training=False)
    else:
        model = T5Gen_Model(args.model_name, data.tokenizer, special_token_list , dropout=0.0,
                            add_special_decoder_token=add_special_decoder_token, is_training=True)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    model.eval()
    print ('Model loaded')

    with torch.no_grad():
        dev_batch_list = data.get_batches(args.number_of_gpu * args.batch_size_per_gpu, mode='dev')
        dev_batch_num_per_epoch = len(dev_batch_list)
        dev_p = progressbar.ProgressBar(dev_batch_num_per_epoch)
        print('Number of evaluation batches is {}'.format(dev_batch_num_per_epoch))
        dev_p.start()
        dev_loss = 0.
        all_dev_result = []
        f = open('sparc_predict/predict_t5_base_with_final_rc_tw_4e-5.txt', 'w')
        # fg = open('gold.txt', 'w')
        from tqdm import tqdm
        for p_dev_idx in tqdm(range(dev_batch_num_per_epoch)):
            dev_p.update(p_dev_idx)
            one_dev_batch = dev_batch_list[p_dev_idx]
            if len(one_dev_batch[0]) == 0:
                break
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels = \
                data.parse_batch_tensor(one_dev_batch)
            if cuda_available:
                dev_batch_src_tensor = dev_batch_src_tensor.to(device)
                dev_batch_src_mask = dev_batch_src_mask.to(device)
                dev_batch_input = dev_batch_input.to(device)
                dev_batch_labels_cuda = dev_batch_labels.to(device)
            one_dev_loss = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels_cuda)
            res = model.batch_generate(dev_batch_src_tensor, dev_batch_src_mask, generate_mode='sqlg', max_decode_len=120)
            for item in range(len(res)):
                # gold_ids = dev_batch_labels[item]
                # if -100 in gold_ids:
                #     i = list(gold_ids.numpy()).index(-100)
                #     gold_ids = gold_ids[:i]
                # gold = model.tokenized_decode(gold_ids)
                # gold = gold.replace('<eos_q>', '')
                all_dev_result.append(res[item])
                f.write(res[item] + '\n')
                # fg.write(gold[:-1] + '\n')


            one_dev_loss = one_dev_loss.mean()
            dev_loss += one_dev_loss.item()
        dev_loss /= dev_batch_num_per_epoch
        f.close()
        # fg.close()
        print('current dev loss is {}'.format(round(dev_loss, 2)))

        # print(all_dev_result)






