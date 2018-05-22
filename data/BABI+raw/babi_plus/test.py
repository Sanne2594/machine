import numpy as np


f = open("dialog-babi-task1-API-calls-trn.txt", "r")
lines = f.readlines()
f.close()
lines1 = [line.split("\t")[0] for line in lines if not line is "\n"]
lines2 = [line.split("\t")[1] for line in lines if not line is "\n"]
lines = [" ".join(line.split(" ")[1:]) for line in lines1]
apis = [line for line in lines if 'api_call' in line]

cuis_slots = {'italian': 1, 'spanish':2,'indian':3, 'french':4, 'british':5, 'korean': 6, 'thai': 7, 'cantonese': 8, 'vietnamese': 9, 'japanese': 10}
loc_slots = {'paris':1, 'bombay':2, 'rome':3, 'london':4, 'madrid':5, 'seoul': 6, 'tokyo' :7, 'beijing': 8, 'bangkok':9, 'hanoi':10}

cuis_list = np.zeros(10)
loc_list = np.zeros(10)

cuis_api = np.zeros(10)
loc_api = np.zeros(10)

for line in lines:
    line = line.split()
    for i,word in enumerate(line):
        if word in cuis_slots:
            cuis_list[cuis_slots[word]-1] += 1
        elif word in loc_slots:
            loc_list[loc_slots[word]-1] += 1

for line in apis:
    line = line.split()
    for i,word in enumerate(line):
        if word in cuis_slots:
            cuis_api[cuis_slots[word]-1] += 1
        elif word in loc_slots:
            loc_api[loc_slots[word]-1] += 1


###########################################################3
## Compute averages and print

rev_cuis = [cuis_list[i]-cuis_api[i] for i,_ in enumerate(cuis_list)]
rev_loc = [loc_list[i]-loc_api[i] for i,_ in enumerate(loc_list)]


cuis_avg = [cuis/cuis_list.sum() for cuis in rev_cuis]
loc_avg = [loc/loc_list.sum() for loc in rev_loc]

print("Cuisines:")
for i in range(10):
    print(list(cuis_slots.keys())[list(cuis_slots.values()).index(i+1)],cuis_avg[i])

print("\nLocations")
for i in range(10):
    print(list(loc_slots.keys())[list(loc_slots.values()).index(i+1)],loc_avg[i])

