import os
from seq2seq.dataset import SourceField, TargetField
import torchtext

import torch
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity

from seq2seq.evaluator import Predictor, Evaluator


def readBabiDialog(fpath,cleanpath,maskpath):
    """
    Reads in babi dialogs from a file.
    Where the final format is: data[dialog_id][turn_id][str] where str is either 'src' OR 'tgt'
    """
    f = open(fpath)
    lines = f.readlines()

    #Initialize dictionaries and their indices
    data = {}
    data2 = {}
    d = 0

    dialog = {}
    diamask = {}
    t= 0
    long = ""

    for line in lines:
        if line != ('\n'):
            if not "\t" in line:
                long = (long + " "+ " ".join(line.split()[1:]))
            else:
                src,tgt,mask = line.split("\t")
                dialog[t] = {'src': (long + " "+ " ".join(src.split()[1:])), 'tgt':tgt.split('\n')[0]}
                diamask[t] = {'src': (long + " "+ " ".join(src.split()[1:])), 'msk':mask.split('\n')[0]}
                long = ""
                t += 1
        else:
            data[d] = dialog
            data2[d] = diamask
            d += 1
            dialog = {}
            diamask = {}
            t = 0
    f.close()

    f = open(cleanpath, "w")
    lines = []

    # create_traindata: For each turn create IN and TARGET
    for d in range(len(data)):
        for t in range(len(data[d])):
            history = ""
            for i in range(0, t):
                history += " ".join([data[d][i]['src'], data[d][i]['tgt']])
            # TODO: Don't add the space when there is no history
            input = history + " " + data[d][t]['src']
            target = data[d][t]['tgt']

            lines += [("\t".join([input, target]))+"\n"]
    f.writelines(lines)
    f.close()

    #Save mask data to file. matching it uniquely with input.
    f = open(maskpath, "w")
    lines = []
    for d in range(len(data2)):
        for t in range(len(data2[d])):
            input = data2[d][t]['src']
            target = data2[d][t]['msk']

            lines += [("\t".join([input, target])) + "\n"]
    f.writelines(lines)
    f.close()

    return



### ### ### General Data
path = "/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/PLUS_data"

# ### ### ### TASK 1: API calls
trainfile = "dialog-babi-task1-API-calls-trn.txt"
testfile = "dialog-babi-task1-API-calls-tst.txt"
devfile = "dialog-babi-task1-API-calls-dev.txt"

dialog_path = "/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/cleaned_data"
mask_path = "/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/mask_data"

train_clean = "task1-trn.txt"
test_clean = "task1-tst.txt"
dev_clean = "task1-dev.txt"

# ### Training files
fpath = os.path.join(path,trainfile)
cpath = os.path.join(dialog_path,train_clean)
mpath = os.path.join(mask_path,train_clean)
readBabiDialog(fpath,cpath,mpath)

# ### Testing files
fpath = os.path.join(path,testfile)
cpath = os.path.join(dialog_path,test_clean)
mpath = os.path.join(mask_path,test_clean)
readBabiDialog(fpath,cpath,mpath)

# ### development files
fpath = os.path.join(path,devfile)
cpath = os.path.join(dialog_path,dev_clean)
mpath = os.path.join(mask_path,dev_clean)
readBabiDialog(fpath,cpath,mpath)



