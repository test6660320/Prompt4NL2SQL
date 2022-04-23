import json

tw_dict = {"0": ["SELECT 改变了AGG", "select change", "select change agg"],
           "1": ["SELECT 删除了列", "select change", "select delete column"],
           "2": ["SELECT 增加了列", "select change", "select add column"],
           "3": ["orderBy改变了", "oder by change", "oder by change"],
           "4": ["改变了LIMIT", "limit change", "limit change"],
           "5": ["FROM 增加了表连接", "from change", "from add table"],
           "6": ["FROM 删除了表连接", "from change", "from delete table"],
           "7": ["改变了except", "except change", "except change"],
           "8": ["WHERE 增加了列条件", "where change", "where add column condition"],
           "9": ["groupby增加了列", "group by change", "group by change"],
           "10": ["SELECT 改变了DISTINCT", "select change", "select change distinct"],
           "11": ["WHERE 增加了条件and/or", "where change", "where add logic condition"],
           "12": ["WHERE 删除了条件", "where change", "where delete condition"],
           "13": ["改变了intersect", "intersect change", "intersect change"],
           "14": ["groupby删除了列", "group by change", "group by delete column"],
           "15": ["改变了union", "union change", "union change"],
           "16": ["Join 增加了条件", "join change", "join add condition"]}

def schema_package(schema):
    schema_p = ''
    for table in schema:
        schema_p += ' <table> ' + table
        columns = schema[table]['columns']
        types = schema[table]['types']
        primary_keys = schema[table]['primary_keys']
        forign_keys = schema[table]['foreign_keys']
        forign_keys_dict = {}
        for key in forign_keys:
            forign_keys_dict[key[0]] = key[1] + ' . ' + key[-1]
        for colunm, type in zip(columns, types):
            schema_p += ' <' + type + '> ' + colunm
            # add primary_keys and forign_keys
            # if colunm in primary_keys:
            #     schema_p += ' <p_key>'
            # if colunm in forign_keys_dict:
            #     schema_p += ' <f_key> ' + forign_keys_dict[colunm]
    schema_p = '<sos_s>' + schema_p + ' <eos_s>'
    return schema_p

def used_schema_package(used_schema):
    text = ''
    for t in used_schema:
        if t == '*':
            continue
        text += ' <table> ' + t
        cols = used_schema[t]
        if cols:
            for col in cols:
                text += ' <column> ' + col
    text = '<sos_rc> ' + text[1:] + ' <eos_rc>'
    return text

# coarse
def turn_change_package(turn_change_index):
    text = ''
    turn_change_index = turn_change_index[-1]
    if turn_change_index:
        turn_change = set()
        for i in turn_change_index:
            turn_change.add(tw_dict[str(i)][1])
        text = ' <\s> '.join(list(turn_change))
    text = '<sos_tw> ' + text + ' <eos_tw>'
    return text

def tokenize_text(tokenizer, text, mode):
    if mode == 'user':
        text = '<sos_u> ' + text + ' <eos_u>'
    elif mode == 'query':
        text = '<sos_q> ' + text + ' <eos_q>'
        # text = text.lower()
    elif mode == 'used_schema':
        text = used_schema_package(text)
    elif mode == 'turn_change':
        text = turn_change_package(text)
    elif mode == 'schema':
        text = schema_package(text)
    else:
        raise Exception('Wrong Mode!!!')
    text = ' '.join(text.split())
    text_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    return text, text_id_list

def process_one_dict(tokenizer, in_dict):
    res_dict = in_dict.copy()
    res_dict["dialogue_session"] = []
    for item in in_dict["dialogue_session"]:
        one_res_dict = {}
        for key in item:
            if key == "turn_num":
                one_res_dict[key] = item[key]
                continue
            elif key == "user":
                mode = 'user'
            elif key == 'query':
                mode = 'query'
            elif key == 'used_schema':
                mode = 'used_schema'
            elif key == 'turn_change_index':
                mode = 'turn_change'
            else:
                raise Exception('Wrong Key!!!')
            id_key = key + '_id_list'
            text, text_id_list = tokenize_text(tokenizer, item[key], mode)
            one_res_dict[key] = text
            one_res_dict[id_key] = text_id_list
        res_dict["dialogue_session"].append(one_res_dict)
    schema = res_dict["schema"]
    text, text_id_list = tokenize_text(tokenizer, schema, 'schema')
    res_dict["schema_package"] = text
    res_dict["schema_id_list"] = text_id_list
    return res_dict


import progressbar
import os


def process_file(path_prefix, file_name, tokenizer, output_path_prefix):
    print('Start processing {}'.format(file_name))
    in_f = path_prefix + file_name
    with open(in_f) as f:
        data = json.load(f)
    data_num = len(data)
    p = progressbar.ProgressBar(data_num)
    p.start()
    res_list = []
    for idx in range(data_num):
        p.update(idx)
        one_res_dict = process_one_dict(tokenizer, data[idx])
        res_list.append(one_res_dict)
    p.finish()
    print('Finish processing {}'.format(file_name))
    save_file = output_path_prefix + r'/' + file_name
    with open(save_file, 'w') as outfile:
        json.dump(res_list, outfile, indent=4)


def process_source_prefix(path_prefix, tokenizer, output_path_prefix):
    file_name_list = os.listdir(path_prefix)
    for name in file_name_list:
        if name != 'tables.json':
            pass
        else:
            continue
        process_file(path_prefix, name, tokenizer, output_path_prefix)


if __name__ == '__main__':
    save_path = r'./'
    tokenizer_path = save_path + r'/tokenizer_with_rc_with_tw'
    print('Loading tokenizer...')
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print('Tokenizer loaded.')
    #
    # # s = "SELECT"
    # # print(tokenizer.tokenize(s))
    # # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)))
    # # s = "select"
    # # print(tokenizer.tokenize(s))
    # # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)))

    path_prefix = r'./data/sparc_data_uncased_with_rc_tw_with_final/'
    save_prefix = r'./data/sparc_tokenized_uncased_with_rc_tw_with_final'
    if os.path.exists(save_prefix):
        pass
    else:  # recursively construct directory
        os.makedirs(save_prefix, exist_ok=True)
    print('Tokenizing sparc Dataset...')
    process_source_prefix(path_prefix, tokenizer, save_prefix)
    print('Dataset Tokenization Finished!')



    # dataset_folder_name_list = ['CamRes676', 'Frames', 'KVRET', 'MetaLWOZ', 'MS_E2E', \
    #                             'Schema_Guided', 'TaskMaster', 'WOZ']
    # for dataset_name in dataset_folder_name_list:
    #     print('Tokenizing {} Dataset...'.format(dataset_name))
    #     path_prefix = source_path_prefix + dataset_name + '/'
    #     process_source_prefix(path_prefix, tokenizer, save_path)
    #     print('{} Dataset Tokenization Finished!'.format(dataset_name))




