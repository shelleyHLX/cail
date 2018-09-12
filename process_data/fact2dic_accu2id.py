# coding: utf-8

import codecs
import json
import pickle
import numpy as np
from multiprocessing import Pool

save_path = '../data/'
y_file_test = 'y_te.npy'
y_file_train = 'y_tr.npy'
y_file_valid = 'y_va.npy'

raw_file_test = "../data_raw/data_test.json"
raw_file_train = '../data_raw/data_train.json'
raw_file_valid = '../data_raw/data_valid.json'

file_fact_test_dict = 'idfact_dict_te.pkl'
file_fact_train_dict = 'idfact_dict_tr.pkl'
file_fact_valid_dict = 'idfact_dict_va.pkl'

# ========== 获得accu的id =========
with open(save_path + 'accu_id.pkl', 'rb') as inp:
    dic_accu_id = pickle.load(inp)

def get_idaccu(accu):
    """获取 accu 所对应的 id."""
    if accu not in dic_accu_id:
        return 1
    else:
        return dic_accu_id[accu]

def get_id4accus(accus):
    """把 accus 转为 对应的 id"""
    ids = list(map(get_idaccu, accus))  # 获取id
    return ids


def read_data(raw_file, file_y_npy, file_fact_dict):
    i = 0
    facts_dict = {}
    accusations_list = []
    f_lines = codecs.open(raw_file, 'r', 'utf-8').readlines()
    print(len(f_lines))

    for line in f_lines:
        if i % 1000 == 0:
            print(i)
        # print(type(line))  # str
        json_line = json.loads(line)
        fact_str = json_line['fact']
        facts_dict[i] = fact_str

        meta = json_line['meta']
        accusations_list.append(meta['accusation'])
        i += 1
        # if i == 50:
        #     break
    p = Pool()
    accusations_list = np.asarray(accusations_list)
    accusations_id_list = np.asarray(list(p.map(get_id4accus, accusations_list)))
    print(accusations_id_list[0:50])
    print(facts_dict[0])

    print('save accusations_id_list', save_path + file_y_npy)
    np.save(save_path + file_y_npy, accusations_id_list)
    print('save facts_dict', save_path + file_fact_dict)
    with open(save_path + file_fact_dict, 'wb') as outp:
        pickle.dump(facts_dict, outp)

if __name__ == '__main__':

    read_data(raw_file_valid, y_file_valid, file_fact_valid_dict)
    read_data(raw_file_train, y_file_train, file_fact_train_dict)
    read_data(raw_file_test, y_file_test, file_fact_test_dict)

