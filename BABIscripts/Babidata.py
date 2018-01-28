import os
from seq2seq.dataset import SourceField, TargetField
import torchtext

import torch
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity

from seq2seq.evaluator import Predictor, Evaluator


def readBabiDialog(fpath,cleanpath):
    """
    Reads in babi dialogs from a file.
    Where the final format is: data[dialog_id][turn_id][str] where str is either 'src' OR 'tgt'

    :param fpath:
    :return data:
    """
    f = open(fpath)
    lines = f.readlines()

    #Initialize dictionaries and their indices
    data = {}
    d = 0

    dialog = {}
    t= 0

    for line in lines:
        if line != ('\n'):
            src,tgt = line.split("\t")
            #src = src.split()[1:].join()
            dialog[t] = {'src':" ".join(src.split()[1:]), 'tgt':tgt.split('\n')[0]}
            t += 1
        else:
            data[d] = dialog
            d += 1
            dialog = {}
            t = 0
    f = open(cleanpath, "w")
    lines = []

    # create_traindata: For each turn create IN and TARGET
    for d in range(len(data)):
        for t in range(len(data[d])):
            history = ""
            for i in range(0, t):
                history += " ".join([data[d][i]['src'], data[d][i]['tgt']])+" "
            # TODO: Don't add the space when there is no history
            input = history + " " + data[d][t]['src']
            target = data[d][t]['tgt']

            lines += [("\t".join([input, target]))+"\n"]
    f.writelines(lines)
    f.close()
    return

def readBabiLines(fpath,line_path,num_path):
    """
    Reads in babi dialogs from a file.
    Where the final format is saved into Field variables.

    :param fpath:
    :return data:
    """
    f = open(fpath)
    lines = f.readlines()
    lines2 = [line for line in lines if line != '\n']
    lines3 = [" ".join(line.split(" ")[1:]) for line in lines2]
    f.close()

    f = open(line_path, 'w')
    # TODO figure out if you need to specifically empty file or if it does so automatically
    # f.seek(0)
    # f.truncate()
    for line in lines3:
        f.write(line)
    f.close()

    f = open(num_path, 'w')
    for line in lines2:
        f.write(line)
    f.close()

    return




### ### ### General Data
path = "/home/sanne/machine/data/BABIraw"
#candidates = "dialog-babi-candidates.txt"
#KB = "dialog-babi-kb-all.txt"


### ### ### TASK 1: API calls
trainfile = "dialog-babi-task1-API-calls-trn.txt"
testfile = "dialog-babi-task1-API-calls-tst.txt"
test_new_entities = "dialog-babi-task1-API-calls-tst-OOV.txt"
devfile = "dialog-babi-task1-API-calls-dev.txt"

line_path = "/home/sanne/machine/data/CLEANED-BABI/babi-line"
dialog_path = "/home/sanne/machine/data/CLEANED-BABI/babi-dialog"
num_path = "/home/sanne/machine/data/CLEANED-BABI/babi-num"
train_clean = "task1-trn.txt"
test_clean = "task1-tst.txt"

### Training files
fpath = os.path.join(path,trainfile)

cpath = os.path.join(line_path,train_clean)
npath = os.path.join(num_path,train_clean)
readBabiLines(fpath,cpath, npath)

cpath = os.path.join(dialog_path,train_clean)
readBabiDialog(fpath,cpath)

### Testing files
fpath = os.path.join(path,testfile)

cpath = os.path.join(line_path,test_clean)
npath = os.path.join(num_path,test_clean)
readBabiLines(fpath,cpath, npath)

cpath = os.path.join(dialog_path,test_clean)
readBabiDialog(fpath,cpath)




### ### ### TASK 2: API refine

# ### ### ### TASK 3: options

# ### ### ### TASK 4: phone adress

# ### ### ### TASK 5: full dialogs

# ### ### ### TASK 6: dstc 2




### ### ### BABI +
path = "/home/sanne/machine/data/BABI+raw/babi_plus"

### ### ### TASK 1: API calls
trainfile = "dialog-babi-task1-API-calls-trn.txt"
testfile = "dialog-babi-task1-API-calls-tst.txt"
#test_new_entities = "dialog-babi-task1-API-calls-tst-OOV.txt"
#devfile = "dialog-babi-task1-API-calls-dev.txt"

line_path = "/home/sanne/machine/data/CLEANED-BABI/babi+line"
dialog_path = "/home/sanne/machine/data/CLEANED-BABI/babi+dialog"
num_path = "/home/sanne/machine/data/CLEANED-BABI/babi+num"
train_clean = "task1-trn.txt"
test_clean = "task1-tst.txt"

### Training files
fpath = os.path.join(path,trainfile)

cpath = os.path.join(line_path,train_clean)
npath = os.path.join(num_path,train_clean)
readBabiLines(fpath,cpath, npath)

cpath = os.path.join(dialog_path,train_clean)
readBabiDialog(fpath,cpath)

### Testing files
fpath = os.path.join(path,testfile)

cpath = os.path.join(line_path,test_clean)
npath = os.path.join(num_path,test_clean)
readBabiLines(fpath,cpath, npath)

cpath = os.path.join(dialog_path,test_clean)
readBabiDialog(fpath,cpath)

