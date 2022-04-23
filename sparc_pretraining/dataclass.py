import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import random
import json
from torch import nn
import os

def map_bool(bool_status):
    if bool_status == 'True': 
        return True
    elif bool_status == 'False': 
        return False
    else:
        raise Exception('Wrong Bool Status')

# format_mapping_dict = {
#     'metalwoz': {'nlu': False, 'bs': False, 'da': False, 'nlg': True},
#     'kvret': {'nlu': False, 'bs': True, 'da': False, 'nlg': True},
#     'woz': {'nlu': False, 'bs': True, 'da': False, 'nlg': True},
#     'camres676': {'nlu': False, 'bs': True, 'da': False, 'nlg': True},
#     'taskmaster': {'nlu': False, 'bs': True, 'da': False, 'nlg': True},
#     'e2e_ms': {'nlu': False, 'bs': True, 'da': True, 'nlg': True},
#     'frames': {'nlu': False, 'bs': True, 'da': True, 'nlg': True},
#     'schema_guided': {'nlu': False, 'bs': True, 'da': True, 'nlg': True}
# }

format_mapping_dict = {
    'sparc': {'nlu': False, 'bs': False, 'da': False, 'nlg': False, 'sqlg': True, 'rcg': True, 'twg': True, 'fug': True}
}

# dataset_name_list = ['e2e_ms', 'metalwoz', 'kvret', 'woz', 'camres676', 'taskmaster', 'frames', 'schema_guided']
dataset_name_list = ['sparc']

class TOD_PRETRAINING_CORPUS:
    def __init__(self, tokenizer, shuffle_mode, dataset_prefix_path, use_nlu, use_bs, use_da, use_nlg,
                 use_sqlg, use_rcg, use_twg, use_fug, max_tgt_len=128):
        self.use_nlu, self.use_bs, self.use_da, self.use_nlg, self.use_sqlg, self.use_rcg, self.use_twg, \
        self.use_fug = map_bool(use_nlu), map_bool(use_bs), map_bool(use_da), map_bool(use_nlg), map_bool(use_sqlg), \
        map_bool(use_rcg), map_bool(use_twg), map_bool(use_fug)
        print ('NLU:{}, DST:{}, POL:{}, NLG:{}, SQLG:{}, RCG:{}, TWG:{}, FUG:{}'.format(use_nlu, use_bs, use_da,
        use_nlg, use_sqlg, use_rcg, use_twg, use_fug))

        print ('Tokenizer Size is %d' % len(tokenizer))
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]

        self.eos_b_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_b>'])[0]
        self.eos_a_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]
        self.eos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_r>'])[0]
        self.eos_d_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_d>'])[0]
        self.eos_q_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_q>'])[0]
        self.eos_rc_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_rc>'])[0]
        self.eos_tw_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_tw>'])[0]
        self.eos_fu_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_fg>'])[0]

        self.sos_b_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_b>'])[0]
        self.sos_a_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]
        self.sos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_r>'])[0]
        self.sos_d_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_d>'])[0]
        self.sos_q_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_q>'])[0]
        self.sos_rc_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_rc>'])[0]
        self.sos_tw_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_tw>'])[0]
        self.sos_fu_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_fg>'])[0]

        self.shuffle_mode = shuffle_mode
        self.max_tgt_len = max_tgt_len

        # construct task-specific prefix
        bs_prefix_text = 'translate dialogue to belief state:'
        self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(bs_prefix_text))
        da_prefix_text = 'translate dialogue to dialogue action:'
        self.da_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(da_prefix_text))
        nlg_prefix_text = 'translate dialogue to system response:'
        self.nlg_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(nlg_prefix_text))
        ic_prefix_text = 'translate dialogue to user intent:'
        self.ic_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ic_prefix_text))
        sqlg_prefix_text = 'translate dialogue to system query:'
        self.sqlg_predix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sqlg_prefix_text))
        rcg_prefix_text = 'translate dialogue to relevant column:'
        self.rcg_predix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(rcg_prefix_text))
        twg_prefix_text = 'translate dialogue to turn switch:'
        self.twg_predix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(twg_prefix_text))
        fug_prefix_text = 'translate dialogue to final utterance:'
        self.fug_predix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(fug_prefix_text))

        self.train_all_dataset_list = self.load_data(dataset_prefix_path, train_test_mode='train', use_bs=self.use_bs,
                                                     use_da=self.use_da, use_nlg=self.use_nlg, use_sqlg=self.use_sqlg,
                                                     use_rcg=self.use_rcg, use_twg=self.use_twg, use_fug=self.use_fug)
        self.dev_all_dataset_list = self.load_data(dataset_prefix_path, train_test_mode='dev', use_bs=self.use_bs,
                                                   use_da=self.use_da, use_nlg=self.use_nlg, use_sqlg=self.use_sqlg,
                                                   use_rcg=self.use_rcg, use_twg=self.use_twg, use_fug=self.use_fug)

        if self.use_nlu == True:
            print ('Add Intent Classification for Pretraining.')
            train_ic_sess_list, dev_ic_sess_list = self.load_train_dev_intent_classification_data(dataset_prefix_path)
        elif self.use_nlu == False:
            print ('Do not Add Intent Classification for Pretraining.')
            train_ic_sess_list, dev_ic_sess_list = [], []
        else:
            raise Exception('Wrong use_Intent_Classification Mode!!!')
        self.train_all_dataset_list += train_ic_sess_list
        self.dev_all_dataset_list += dev_ic_sess_list
        train_session_num, dev_session_num = len(self.train_all_dataset_list), len(self.dev_all_dataset_list)

        self.train_data_list = self.shuffle_train_data()
        self.dev_data_list = self.flatten_data(self.dev_all_dataset_list, mode='dev_set')
        print ('train session number is {}, train turn number is {}, train turn number per session {}'.format(train_session_num, len(self.train_data_list), round(len(self.train_data_list)/train_session_num, 2)))
        print ('dev session number is {}, dev turn numbder is {}, dev turn number per session {}'.format(dev_session_num, len(self.dev_data_list), round(len(self.dev_data_list)/dev_session_num, 2)))
        self.train_num, self.dev_num = len(self.train_data_list), len(self.dev_data_list)

    def load_data(self, dataset_prefix_path, train_test_mode, use_bs, use_da, use_nlg, use_sqlg, use_rcg, use_twg, use_fug):
        all_dataset_list = []
        for name in dataset_name_list:
            one_dataset_list = self.parse_one_dataset(dataset_prefix_path, name, train_test_mode,
                                                      use_bs, use_da, use_nlg, use_sqlg, use_rcg, use_twg, use_fug)
            if len(one_dataset_list) > 0:
                all_dataset_list += one_dataset_list
        # print('')
        return all_dataset_list

    def parse_one_dataset(self, dataset_prefix_path, data_set_name, train_test_mode,
                          use_bs, use_da, use_nlg, use_sqlg, use_rcg, use_twg, use_fug):
        assert train_test_mode in ['train', 'dev']
        # train_test_mode: 'train' or 'test'
        bs_exist, da_exist, nlg_exist, sqlg_exist, rcg_exist, twg_exist, fug_exist = format_mapping_dict[data_set_name]['bs'], \
        format_mapping_dict[data_set_name]['da'], format_mapping_dict[data_set_name]['nlg'], \
        format_mapping_dict[data_set_name]['sqlg'], format_mapping_dict[data_set_name]['rcg'], \
        format_mapping_dict[data_set_name]['twg'], format_mapping_dict[data_set_name]['fug']

        dataset_path = dataset_prefix_path + train_test_mode + '.json'

        print ('Loading data from {}'.format(dataset_path))
        with open(dataset_path) as f:
            data = json.load(f) 

        all_sess_list = []
        for one_sess in data:
            schema_id_list = one_sess['schema_id_list'] # this is the schema ids
            dial_sess_list = one_sess["dialogue_session"] # this list contains all turns from on session
            one_sess_list = [] 
            # one_sess_list is a list of turns
            # each turn is list of tuple pairs
            previous_context = []
            previous_utterance = []

            # final
            final_turn = dial_sess_list[-1]
            final_user_input = final_turn['user_id_list']
            final_sys_query = final_turn['query_id_list']
            final_sqlg_input = final_user_input + schema_id_list
            final_sqlg_input = self.sqlg_predix_id + [self.sos_context_token_id] + final_sqlg_input[-900:] + \
                         [self.eos_context_token_id]
            final_sqlg_output = final_sys_query[:-1][:self.max_tgt_len] + [self.eos_q_token_id]
            final_turn_list = [(final_sqlg_input, final_sqlg_output)]

            turn_num = len(dial_sess_list) - 1
            for turn_id in range(turn_num):
                curr_turn = dial_sess_list[turn_id]
                curr_turn_list = []
                # this is a list of tuple pair (src, tgt)
                # [(nlg_input, nlg_output), (bs_input, bs_output), (da_input, da_output)]
                curr_user_input = curr_turn['user_id_list']
                curr_sys_query = curr_turn['query_id_list']

                # ----------------------------------------------------------- #
                if use_sqlg and sqlg_exist:
                    sqlg_input = previous_context + curr_user_input
                    sqlg_input = sqlg_input + schema_id_list
                    sqlg_input = self.sqlg_predix_id + [self.sos_context_token_id] + sqlg_input[-900:] + \
                               [self.eos_context_token_id]
                    sqlg_output = curr_sys_query[:-1][:self.max_tgt_len] + [self.eos_q_token_id]
                    curr_turn_list.append((sqlg_input, sqlg_output))

                # with previous sql
                if use_rcg and rcg_exist:
                    rcg_input = previous_context + curr_user_input
                    rcg_input = rcg_input + schema_id_list
                    rcg_input = self.rcg_predix_id + [self.sos_context_token_id] + rcg_input[-900:] + \
                    [self.eos_context_token_id]
                    curr_rc = curr_turn['used_schema_id_list']
                    rcg_output = curr_rc[:-1][:self.max_tgt_len] + [self.eos_rc_token_id]
                    curr_turn_list.append((rcg_input, rcg_output))

                # with previous sql
                if use_twg and twg_exist:
                    twg_input = previous_context + curr_user_input
                    twg_input = twg_input + schema_id_list
                    twg_input = self.twg_predix_id + [self.sos_context_token_id] + twg_input[-900:] + \
                                [self.eos_context_token_id]
                    curr_tw = curr_turn['turn_change_index_id_list']
                    if len(curr_tw) > 2:
                        twg_output = curr_tw[:-1][:self.max_tgt_len] + [self.eos_tw_token_id]
                        curr_turn_list.append((twg_input, twg_output))

                if use_nlg and nlg_exist: # adding nlg data into pre-training procedure
                    # construct nlg_input, nlg_output
                    nlg_input = previous_context + curr_user_input
                    nlg_input = self.nlg_prefix_id + [self.sos_context_token_id] + \
                    nlg_input[-900:] + [self.eos_context_token_id]
                    curr_sys_resp = curr_turn['resp_id_list']
                    nlg_output = curr_sys_resp[:-1][:self.max_tgt_len] + [self.eos_r_token_id] # constrain the maximum tgt len
                    curr_turn_list.append((nlg_input, nlg_output))

                if use_bs and bs_exist:
                    bs_input = previous_context + curr_user_input
                    bs_input = self.bs_prefix_id + [self.sos_context_token_id] + bs_input[-900:] + \
                    [self.eos_context_token_id]
                    curr_bspn = curr_turn['bspn_id_list']
                    bs_output = curr_bspn[:-1][:self.max_tgt_len] + [self.eos_b_token_id]
                    curr_turn_list.append((bs_input, bs_output))

                if use_da and da_exist:
                    da_input = previous_context + curr_user_input
                    da_input = self.da_prefix_id + [self.sos_context_token_id] + da_input[-900:] + \
                        [self.eos_context_token_id]
                    curr_aspn = curr_turn['aspn_id_list']
                    da_output = curr_aspn[:-1][:self.max_tgt_len] + [self.eos_a_token_id]
                    curr_turn_list.append((da_input, da_output))

                if len(curr_turn_list) > 0:
                    one_sess_list.append(curr_turn_list)
                # update previous context
                previous_utterance = previous_utterance + curr_user_input
                previous_context = previous_context + curr_user_input + curr_sys_query

            if use_fug and fug_exist:
                fug_input = previous_utterance
                fug_input = self.fug_predix_id + [self.sos_context_token_id] + fug_input[-900:] + \
                           [self.eos_context_token_id]
                fug_output = [self.sos_fu_token_id] + final_user_input[1:-1] + [self.eos_fu_token_id]
                fug_output = fug_output[:-1][:self.max_tgt_len] + [self.eos_fu_token_id]
                one_sess_list.append([(fug_input, fug_output)])
            if use_sqlg and sqlg_exist:
                one_sess_list.append(final_turn_list)

            if len(one_sess_list) > 0:
                all_sess_list.append(one_sess_list)
        return all_sess_list

    def load_intent_classification_data(self, path):
        
        '''
            we treat each instance as a session, each data instance is treated as one session with one turn.
            so all instances have the following format
            [
                [
                    [(ic_src, ic_tgt)]
                ],

                [
                    [(ic_src, ic_tgt)]
                ],

                ...
            ]
        '''
        print ('Loading data from {}'.format(path))
        all_sess_list = []
        with open(path) as f:
            data = json.load(f) 
        for one_dict in data:
            one_intent_input = self.ic_prefix_id + [self.sos_context_token_id] + \
                one_dict['user_id_list'][-900:] + [self.eos_context_token_id]
            one_intent_output = one_dict['intent_id_list']
            one_turn = [(one_intent_input, one_intent_output)]
            one_sess = [one_turn]
            all_sess_list.append(one_sess)
        return all_sess_list

    def load_train_dev_intent_classification_data(self, dataset_prefix_path):
        train_path = dataset_prefix_path + '/train_intent_classification.json'
        train_ic_sess_list = self.load_intent_classification_data(train_path)
        dev_path = dataset_prefix_path + '/test_intent_classification.json'
        dev_ic_sess_list = self.load_intent_classification_data(dev_path)
        return train_ic_sess_list, dev_ic_sess_list

    def shuffle_train_data(self):
        return self.flatten_data(self.train_all_dataset_list, mode='train_set')

    def flatten_data(self, all_dataset_session_list, mode):
        '''
            all_dataset_session_list: 
                contains all sessions from all datasets
                each session contains multiple turns
                each turn is a list which has a format ranging from 
                    [(nlg_input, nlg_output)], 
                    [(nlg_input, nlg_output), (bs_input, bs_output)],
                    [(nlg_input, nlg_output), (bs_input, bs_output), (da_input, da_output)]
        '''
        flatten_data_list = []
        if mode == 'train_set':
            if self.shuffle_mode == 'session_level':
                tmp_session_list = all_dataset_session_list.copy()
                random.shuffle(tmp_session_list)
                for one_session in tmp_session_list:
                    for one_turn in one_session:
                        for one_tuple in one_turn:
                            flatten_data_list.append(one_tuple)
            elif self.shuffle_mode == 'turn_level':
                for one_session in all_dataset_session_list:
                    for one_turn in one_session:
                        for one_tuple in one_turn:
                            flatten_data_list.append(one_tuple)
                random.shuffle(flatten_data_list)
            else:
                raise Exception('Wrong Shuffle Mode!!!')
        elif mode == 'dev_set':
            for one_session in all_dataset_session_list:
                for one_turn in one_session:
                    for one_tuple in one_turn:
                        flatten_data_list.append(one_tuple)
        else:
            raise Exception()
        return flatten_data_list

    def get_batches(self, batch_size, mode):
        #batch_size = self.cfg.batch_size
        batch_list = []
        if mode == 'train':
            self.train_data_list = self.shuffle_train_data()
            all_data_list = self.train_data_list
        elif mode == 'dev':
            all_data_list = self.dev_data_list
        else:
            raise Exception('Wrong Mode!!!')

        all_input_data_list, all_output_data_list = [], []
        for inp, oup in all_data_list:
            all_input_data_list.append(inp)
            all_output_data_list.append(oup)

        data_num = len(all_input_data_list)
        batch_num = int(data_num/batch_size) + 1

        for i in range(batch_num):
            start_idx, end_idx = i*batch_size, (i+1)*batch_size
            if start_idx > data_num - 1:
                break
            end_idx = min(end_idx, data_num - 1)
            one_input_batch_list, one_output_batch_list = [], []
            for idx in range(start_idx, end_idx):
                one_input_batch_list.append(all_input_data_list[idx])
                one_output_batch_list.append(all_output_data_list[idx])
            one_batch = [one_input_batch_list, one_output_batch_list]
            batch_list.append(one_batch)
        out_str = 'Overall Number of datapoints is ' + str(data_num) + \
        ' Number of ' + mode + ' batches is ' + str(len(batch_list))
        print (out_str)
        return batch_list

    def build_iterator(self, batch_size, mode):
        batch_list = self.get_batches(batch_size, mode)
        for i, batch in enumerate(batch_list):
            yield batch

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list)
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels

