# coding: utf-8

accu_file = '../data_raw/accu.txt'
file_data = '../data/'

import pickle
import codecs

def accu4id(file_name):
    accu_id = dict()
    id_accu = dict()
    i = 0
    rf = codecs.open(file_name, 'r', 'utf-8')
    for line in rf.readlines():
        line = line.replace('\n', '').replace('\r', '')
        accu_id[line] = i
        id_accu[i] = line
        i += 1
    print(accu_id)
    with open(file_data + 'accu_id.pkl', 'wb') as outp:
        pickle.dump(accu_id, outp)
    with open(file_data + 'id_accu.pkl', 'wb') as outp:
        pickle.dump(id_accu, outp)


if __name__ == '__main__':
    accu4id(accu_file)
