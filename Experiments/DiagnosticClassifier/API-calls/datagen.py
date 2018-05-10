import os
import numpy as np

data_in = "../../../data/CLEANED-BABI/babi+dialog/"
in_file = "task1-trn.txt"

path = "Data/"
cuis_file = "cuisine_masks.txt"
loc_file = "location_mask.txt"
size_file = "party_size_masks.txt"
price_file = "price_range_masks.txt"

def strip_API(fpath):
    f = open(fpath, 'r')
    lines = f.readlines()
    f.close()

    lines_api = [line for line in lines if 'api_call' in line]
    return lines_api


fpath = os.path.join(data_in,in_file)
api_only = strip_API(fpath)
cuis_slots = {'italian': 1, 'spanish':2,'indian':3, 'french':4, 'british':5, 'korean': 6, 'thai': 7, 'cantonese': 8, 'vietnamese': 9, 'japanese': 10}
loc_slots = {'paris':1, 'bombay':2, 'rome':3, 'london':4, 'madrid':5}
size_slots = {'four':1, 'six':2, 'eight':3, 'two':4}
price_slots = {'moderate':1, 'cheap':2, 'expensive':3}

cuis_list = []
loc_list = []
size_list = []
price_list = []
for line in api_only:
    sent = line.split('\t')[0].split()

    cuis_mask = np.zeros(len(sent))
    loc_mask = np.zeros(len(sent))
    size_mask = np.zeros(len(sent))
    price_mask = np.zeros(len(sent))

    for i,word in enumerate(sent):
        if word in cuis_slots:
            cuis_mask[i:] = cuis_slots[word]
        elif word in loc_slots:
            loc_mask[i:] = loc_slots[word]
        elif word in size_slots:
            size_mask[i:] = size_slots[word]
        elif word in price_slots:
            price_mask[i:] = price_slots[word]

    cuis_line = " ".join(sent)+ "\t" + ' '.join([str(i) for i in cuis_mask]) + '\n'
    loc_line = " ".join(sent) + "\t" + ' '.join([str(i) for i in loc_mask]) + '\n'
    size_line = " ".join(sent) + "\t" + ' '.join([str(i) for i in size_mask]) + '\n'
    price_line = " ".join(sent) + "\t" + ' '.join([str(i) for i in price_mask]) + '\n'

    cuis_list.append(cuis_line)
    loc_list.append(loc_line)
    size_list.append(size_line)
    price_list.append(price_line)

cpath = os.path.join(path,cuis_file)
f = open(cpath, 'w')
f.writelines(cuis_list)
f.close()

cpath = os.path.join(path,loc_file)
f = open(cpath, 'w')
f.writelines(loc_list)
f.close()

cpath = os.path.join(path,size_file)
f = open(cpath, 'w')
f.writelines(size_list)
f.close()

cpath = os.path.join(path,price_file)
f = open(cpath, 'w')
f.writelines(price_list)
f.close()


