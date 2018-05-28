import os

data_in = "../../../data/CLEANED-BABI/babi-dialog/"
in_file = "task1-trn.txt"
fpath = os.path.join(data_in,in_file)

path = "Data-normal/"
out_file = "trn_aug.txt"


f = open(fpath, 'r')
lines = f.readlines()
f.close()

newlines = []
for line in lines:
    human,bot = line.split("\t")
    newline = human +  " ok let me look into some options <silence>\tapi_call this will be discarded"
    newlines.append(newline)

fpath = os.path.join(path,out_file)
f = open(fpath, "r")
f.writelines(newlines)
f.close()