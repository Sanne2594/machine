import os
import argparse
import logging
import torch
import numpy as np
import torchtext
from torch.autograd import Variable

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.util.checkpoint import Checkpoint


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--mask_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)

opt = parser.parse_args()

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

############################################################################
# Prepare dataset and vocabularies
src = SourceField()
msk = TargetField()
src.vocab = input_vocab
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len

#TODO: nadenken over het format waarin mask wordt uitgelezen
# generate test set
data = torchtext.data.TabularDataset(
    path=opt.mask_data, format='tsv',
    fields=[('src', src), ('msk', msk)],
    filter_pred=len_filter
)

src_vocab = data.fields['src'].vocab #Somehow get this


#Read in words line by line, word by word
loc = opt.mask_data
f = open(loc, 'r')
lines = f.readlines()
f.close()
# TODO: create an array size lines to save results in?
statistic = []

for line in lines:
    src_sentence,mask = line.split("\t")
    src_seq = src_sentence.split()

#    i = 0
#    for item in src_seq:
    src_id_seq = Variable(torch.LongTensor(src_vocab.stoi[src_seq]), volatile=True).view(1, -1)
    if torch.cuda.is_available():
        src_id_seq = src_id_seq.cuda()
    output,hidden = seq2seq.encoder(src_id_seq)
    mask_one = mask[i]
    #i += 1
    #TODO: Save the statistic, matched with the mask instance {0,1} of that word to a file.
    hidden_temp = [str(hid) for hid in hidden]
    hidden_string = " ".join(hidden_temp)
    statistic.append("\t".join([hidden_string,mask_one]))

# This will be a big dataset.                          1000*6*20 = 120 000
#  formulae: num dialogs*average dialog length * average word per sentence
result_file = "hidden_and_mask.txt"
m = open(result_file)
m.writelines(statistic)
m.close()