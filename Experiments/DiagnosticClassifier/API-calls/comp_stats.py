import os
import numpy as np

data_in = "../../../data/CLEANED-BABI/babi+dialog/"
in_file = "task1-trn.txt"


############################
## Get APi-call lines

fpath = os.path.join(data_in,in_file)
f = open(fpath, 'r')
lines = f.readlines()
f.close()

lines_api = [line for line in lines if 'api_call' in line]


###########################################3
## Get occurences statistic

cuis_slots = {'italian': 1, 'spanish':2,'indian':3, 'french':4, 'british':5, 'korean': 6, 'thai': 7, 'cantonese': 8, 'vietnamese': 9, 'japanese': 10}
loc_slots = {'paris':1, 'bombay':2, 'rome':3, 'london':4, 'madrid':5, 'seoul': 6, 'tokyo' :7, 'beijing': 8, 'bangkok':9, 'hanoi':10}
size_slots = {'four':1, 'six':2, 'eight':3, 'two':4}
price_slots = {'moderate':1, 'cheap':2, 'expensive':3}

cuis_list = np.zeros(10)
loc_list = np.zeros(10)
size_list = np.zeros(4)
price_list = np.zeros(3)

for line in lines_api:
    sent = line.split('\t')[0].split()
    for i,word in enumerate(sent):
        if word in cuis_slots:
            cuis_list[cuis_slots[word]-1] += 1
        elif word in loc_slots:
            loc_list[loc_slots[word]-1] += 1
        elif word in size_slots:
            size_list[size_slots[word]-1] += 1
        elif word in price_slots:
            price_list[price_slots[word]-1] += 1


###########################################################3
## Compute averages and print

cuis_avg = [cuis/cuis_list.sum() for cuis in cuis_list]
loc_avg = [loc/loc_list.sum() for loc in loc_list]
size_avg = [size/size_list.sum() for size in size_list]
price_avg = [price/price_list.sum() for price in price_list]

print("Cuisines:")
for i in range(10):
    print(list(cuis_slots.keys())[list(cuis_slots.values()).index(i+1)],cuis_avg[i])

print("\nLocations")
for i in range(10):
    print(list(loc_slots.keys())[list(loc_slots.values()).index(i+1)],loc_avg[i])

print("\nParty Size")
for i in range(4):
    print(list(size_slots.keys())[list(size_slots.values()).index(i+1)],size_avg[i])

print("\nPrice Range")
for i in range(3):
    print(list(price_slots.keys())[list(price_slots.values()).index(i+1)], price_avg[i])