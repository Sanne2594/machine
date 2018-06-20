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
parser.add_argument('--predict', action='store_true')
parser.add_argument('--attviz', help='Give path to image folder')
parser.add_argument('--print_wrong', type=float, help='Threshold accuracy per sequence')
parser.add_argument('--select_eval', action='store_true', help='Evaluate only tokens whose target is now "<unk>", "api_call" or "<eos>"')
parser.add_argument('--hidviz', help='Give path to image folder')

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

# Prepare loss
#TODO Introduce cross entropy
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

#Check if print_wrong is true
if (opt.print_wrong):
    if (not opt.batch_size is 1):
        print("Warning: Expected batch size 1, got", opt.batch_size)
        opt.batch_size=1
#################################################################################
# Evaluate model on test set

evaluator = Evaluator(loss=loss, batch_size=opt.batch_size)
if (opt.print_wrong):
    loss, accuracy, seq_accuracy = evaluator.evaluate(seq2seq, test, threshold=opt.print_wrong, select_eval=opt.select_eval)
else:
    loss, accuracy, seq_accuracy = evaluator.evaluate(seq2seq, test, select_eval=opt.select_eval)

print("\nLoss: %f, Word accuracy: %f, Sequence accuracy: %f" % (loss, accuracy, seq_accuracy))

raw_input()

#########################################
#Predict sentences

if(opt.predict):
    if("+dialog" in opt.test_data):
        path = "data/CLEANED-BABI/sample-dialogs/dialog-plus.txt"
        print("\nPredicting for Babi plus Dialogs\n")
    elif("plus-dialog" in opt.test_data):
        path = "data/CLEANED-BABI/sample-dialogs/dialog-plus.txt"
        print("\nPredicting for Babi plus Dialogs\n")
    elif("-dialog" in opt.test_data):
        path = "data/CLEANED-BABI/sample-dialogs/dialog.txt"
        print("\nPredicting for Babi Dialogs\n")
    else:
        print("Couldn't find matching sample dialog")
        print(opt.test_data)

    path = opt.test_data

    f = open(path,"r")
    lines = f.readlines()
    f.close()

    predictor = Predictor(seq2seq, input_vocab, output_vocab)

    for line in lines:
        input, exp = line.split("\t")
        output = predictor.predict(input.split())
        print("\n", input)
        print("Output:", " ".join(output), 'expected output', exp)
        raw_input()

if(opt.attviz):
    if(not opt.predict):
        path = "data/CLEANED-BABI/sample-dialogs/dialog.txt"
        if ("shorter.txt" in opt.test_data):
            path = "shorter.txt"
            print("hacked the system: Train plus dialog")
        elif ("+dialog" in opt.test_data):
            path = "data/CLEANED-BABI/sample-dialogs/dialog-plus.txt"
            print("\nPredicting for Babi plus Dialogs\n")
        elif ("plus-dialog" in opt.test_data):
            path = "data/CLEANED-BABI/sample-dialogs/dialog-plus.txt"
            print("\nPredicting for Babi plus Dialogs\n")
        elif ("-dialog" in opt.test_data):
            path = "data/CLEANED-BABI/sample-dialogs/dialog.txt"
            print("\nPredicting for Babi Dialogs\n")
        else:
            print("Couldn't find matching sample dialog")
            print(opt.test_data)
        f = open(path, "r")
        lines = f.readlines()
        f.close()

    #At this point lines consists of dialog input output pairs.

    #lines2 = [line for line in lines if 'api_call' in line]
    lines2 = lines
    count = 0

    for line in lines2:
        input,exp = line.split("\t")
        #output = predictor.predict(input.split())
        if torch.cuda.is_available():
            src_id_seq = Variable(torch.cuda.LongTensor([input_vocab.stoi[tok] for tok in input.split()]),
                                  volatile=True).view(1, -1)
        else:
            src_id_seq = Variable(torch.LongTensor([input_vocab.stoi[tok] for tok in input.split()]),
                                  volatile=True).view(1, -1)

        decoder_outputs, decoder_hidden, other = seq2seq(src_id_seq, [len(input.split())])
        length = other['length'][0]

        attentions = np.empty([length,len(input.split())])
        for oi in range(length):
            sequence =  other["attention_score"][oi]
            attentions[oi, :] = sequence[0][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [output_vocab.itos[tok] for tok in tgt_id_seq]

        # fig = showAttention(input.split(),tgt_seq,attentions)
        # fig_loc = opt.attviz + "attn" + str(count) + ".png"
        out_loc = opt.attviz + "attn" + str(count) + ".txt"
        count += 1
        # fig.savefig(fig_loc)
        saveAttention(input.split(),tgt_seq,attentions,out_loc)

if opt.hidviz:
    if(not opt.predict and not opt.attvizz):
        path = "data/CLEANED-BABI/sample-dialogs/dialog.txt"
        if ("shorter.txt" in opt.test_data):
            path = "shorter.txt"
            print("hacked the system: Train plus dialog")
        elif ("+dialog" in opt.test_data):
            path = "data/CLEANED-BABI/sample-dialogs/dialog-plus.txt"
            print("\nPredicting for Babi plus Dialogs\n")
        elif ("plus-dialog" in opt.test_data):
            path = "data/CLEANED-BABI/sample-dialogs/dialog-plus.txt"
            print("\nPredicting for Babi plus Dialogs\n")
        elif ("-dialog" in opt.test_data):
            path = "data/CLEANED-BABI/sample-dialogs/dialog.txt"
            print("\nPredicting for Babi Dialogs\n")
        else:
            print("Couldn't find matching sample dialog")
            print(opt.test_data)
        f = open(path, "r")
        lines = f.readlines()
        f.close()

    for line in lines2:
        input, exp = line.split("\t")
        # output = predictor.predict(input.split())
        if torch.cuda.is_available():
            src_id_seq = Variable(torch.cuda.LongTensor([input_vocab.stoi[tok] for tok in input.split()]),
                                  volatile=True).view(1, -1)
        else:
            src_id_seq = Variable(torch.LongTensor([input_vocab.stoi[tok] for tok in input.split()]),
                                  volatile=True).view(1, -1)

        #Somehow tell the forward to roll out and print or return every encoder attention vector
        decoder_outputs, decoder_hidden, other = seq2seq(src_id_seq, [len(input.split())])

        #TODO: visualize decoder_hidden