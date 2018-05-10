import os
import argparse
import logging
import torch
import matplotlib
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

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return fig

def saveAttention(input_sentence, output_words, attentions, file):
    # open file.
    f = open(file, "w")
    #Write input sentence, \n output words, \n, attentions (check of dat goed gaat), /n/n??
    f.write(" ".join(input_sentence)+"\n")
    f.write(" ".join(output_words)+"\n")
    f.write(" ".join([str(np.ndim(attentions)), str(len(attentions)),str(len(attentions[0])) ])+"\n")
    attentions.tofile(f," ")
    f.close()


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_data', help='Path to test data')
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

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(vis_mat, cmap='Greys')
ax.set_xticklabels([' '] + input_words, rotation=90)
ax.set_yticklabels([' '] + output_words)

# Show label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()


