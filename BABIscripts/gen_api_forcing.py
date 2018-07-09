import os
import numpy as np

#cuisine_mask = read in from a file, is per api_call
mask_path = "../Experiments/DiagnosticClassifier/API-calls/Data-babi-masks"
#TODO: updaten naar de goede masks.

cuis_file = "tst_cuisine_masks.txt"
fpath = os.path.join(mask_path,cuis_file)
f = open(fpath, 'r')
lines = f.readlines()
f.close()
cuis_mask = lines

loc_file = "tst_location_masks.txt"
fpath = os.path.join(mask_path,loc_file)
f = open(fpath, 'r')
lines = f.readlines()
f.close()
loc_mask = lines

size_file = "tst_party_size_masks.txt"
fpath = os.path.join(mask_path,size_file)
f = open(fpath, 'r')
lines = f.readlines()
f.close()
size_mask = lines

price_file = "tst_price_range_masks.txt"
fpath = os.path.join(mask_path,price_file)
f = open(fpath, 'r')
lines = f.readlines()
f.close()
price_mask = lines

cuis_slots = {1:'italian', 2:'spanish',3:'indian', 4:'french', 5:'british', 6:'korean', 7:'thai', 8:'cantonese', 9:'vietnamese', 0:'japanese',-1:"<unk>"}
loc_slots = {1:'paris', 2:'bombay', 3:'rome', 4:'london', 5:'madrid', 6:'seoul', 7:'tokyo' , 8:'beijing', 9:'bangkok', 0:'hanoi',-1:"<unk>"}
size_slots = {1:'four', 2:'six', 3:'eight', 0:'two',-1:"<unk>"}
price_slots = {1:'moderate', 2:'cheap', 0:'expensive',-1:"<unk>"}


#Start if actual dialogs reading in.
data_in = "../data/CLEANED-BABI/babi-dialog/"
in_file = "task1-tst.txt"
fpath = os.path.join(data_in,in_file)

f = open(fpath, 'r')
lines = f.readlines()
f.close()

i=0
newlines = []
for line in lines:
    human, bot = line.split("\t")

    if "api_call" in line:
        cuis =  cuis_slots[np.fromstring(cuis_mask[i].split("\t")[1],dtype="float", sep=" ")[-1]]
        loc=     loc_slots[np.fromstring( loc_mask[i].split("\t")[1],dtype="float", sep=" ")[-1]]
        size =  size_slots[np.fromstring(size_mask[i].split("\t")[1],dtype="float", sep=" ")[-1]]
        price=price_slots[np.fromstring(price_mask[i].split("\t")[1],dtype="float", sep=" ")[-1]]
        i = i + 1
    else:
        cuis = cuis_slots[np.fromstring(cuis_mask[i].split("\t")[1],dtype="float", sep=" ")[len(human.split())]]
        loc = loc_slots[np.fromstring(loc_mask[i].split("\t")[1],dtype="float", sep=" ")[len(human.split())]]
        size = size_slots[np.fromstring(size_mask[i].split("\t")[1],dtype="float", sep=" ")[len(human.split())]]
        price = price_slots[np.fromstring(price_mask[i].split("\t")[1],dtype="float", sep=" ")[len(human.split())]]
    api_call = "\t api_call " + " ".join([ cuis , loc,size,price  ]) + "\n"
    newline = human +  " ok let me look into some options <silence>" + api_call
    newlines.append(newline)


path = "../Experiments/DiagnosticClassifier/API-calls/Data-api-forcing/"
out_file = "min_tst_aug.txt"
fpath = os.path.join(path,out_file)
f = open(fpath, "w")
f.writelines(newlines)
f.close()