import os
import argparse
import logging
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torchtext
from torch.autograd import Variable

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.util.checkpoint import Checkpoint

import pickle

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_data', help='Path to test data')
parser.add_argument('--output_dir', help='Give the path where output should be generated')
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
# Prepare dataset and loss
src = SourceField()
tgt = TargetField()
src.vocab = input_vocab
tgt.vocab = output_vocab
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate test set
test = torchtext.data.TabularDataset(
    path=opt.test_data, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

#################################################################################
# Evaluate model on test set

def filter_matrix(matrix, input_words, output_words):
    return matrix, input_words, output_words

evaluator = Evaluator(batch_size=opt.batch_size)

cooccurrences = evaluator.get_cooccurence_matrix(seq2seq, test)

output_indices = cooccurrences.keys()
input_indices = list(set([word for word in [val.keys() for val in cooccurrences.values()]][0]))
output_words = [output_vocab.itos[index] for index in output_indices]
input_words = [input_vocab.itos[index] for index in input_indices]

vis_mat = np.zeros((len(output_words), len(input_words)))

for i, output_index in enumerate(output_indices):
    sum_vals = sum(cooccurrences[output_index].values())
    for j, input_index in enumerate(input_indices):
        vis_mat[i,j] = cooccurrences[output_index].get(input_index, 0)/sum_vals

#print maken van vis_mat, input_words, output_words Met pickle

pickle.dump(vis_mat,  open( "vismat.p", "wb" ) )
pickle.dump(input_words,  open( "input.p", "wb" ) )
pickle.dump(output_words,  open( "output.p", "wb" ) )

# vis_mat = pickle.load(open("vismat.p", "rb"))
# input_words = pickle.load(open("input.p", "rb"))
# output_words = pickle.load(open("output.p", "rb"))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(vis_mat, cmap='Greys')
# ax.set_xticklabels([' '] + input_words, rotation=90)
# ax.set_yticklabels([' '] + output_words)
#
# # Show label at every tick
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# #plt.show()
#
# DefaultSize = fig.get_size_inches()
# fig.set_size_inches(DefaultSize[0] * 2, DefaultSize[1] * 2)
#
# out_loc = opt.output_dir + "coocur.png"
# fig.savefig(out_loc)
