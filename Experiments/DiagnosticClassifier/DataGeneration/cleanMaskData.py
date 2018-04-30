import os
from seq2seq.dataset import SourceField, TargetField
import torchtext
import ast
import numpy as np

import torch
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity

from seq2seq.evaluator import Predictor, Evaluator


def readBabiDialog(fpath,maskpath):
    """
    Reads in babi dialogs from a file.
    Where the final format is: data[dialog_id][turn_id][str] where str is either 'src' OR 'tgt'
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
            src,tgt,mask = line.split("\t")
            dialog[t] = {'src': (" ".join(src.split()[1:])), 'tgt':tgt, 'msk':mask.split('\n')[0]}
            t += 1
        else:
            data[d] = dialog
            d += 1
            dialog = {}
            t = 0
    f.close()

    f = open(maskpath, "w")
    lines = []

    # create_traindata: For each turn create IN and TARGET
    for d in range(len(data)):
        for t in range(len(data[d])):
            history = ""
            maskhist = []
            for i in range(0, t):
                history += " ".join([data[d][i]['src'], data[d][i]['tgt']])
                maskhist.append(ast.literal_eval(data[d][i]['msk']))
                maskhist.append([0]*len(data[d][i]['tgt'].split()))
            # TODO: Don't add the space when there is no history
            input = history + " " + data[d][t]['src']
            #target = data[d][t]['tgt']
            maskhist.append(ast.literal_eval(data[d][t]['msk']) )
            maskput = str(np.concatenate(maskhist))
            maskput = " ".join(maskput.split("\n"))
            lines += [("\t".join([input, maskput]))+"\n"]
    f.writelines(lines)
    f.close()
    return



### ### ### General Data
path = "/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/PLUS_data"

# ### ### ### TASK 1: API calls
# trainfile = "dialog-babi-task1-API-calls-trn.txt"
# testfile = "dialog-babi-task1-API-calls-tst.txt"
# devfile = "dialog-babi-task1-API-calls-dev.txt"
trainfile = "temp_noerr.txt"

mask_path = "/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/mask_data"

# train_clean = "task1-trn.txt"
# test_clean = "task1-tst.txt"
# dev_clean = "task1-dev.txt"
train_clean = "temp_noerr.txt"

# ### Training files
fpath = os.path.join(path,trainfile)
mpath = os.path.join(mask_path,train_clean)
readBabiDialog(fpath,mpath)

# # ### Testing files
# fpath = os.path.join(path,testfile)
# mpath = os.path.join(mask_path,test_clean)
# readBabiDialog(fpath,mpath)
#
# # ### development files
# fpath = os.path.join(path,devfile)
# mpath = os.path.join(mask_path,dev_clean)
# readBabiDialog(fpath,mpath)



