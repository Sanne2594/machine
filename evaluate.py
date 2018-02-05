import os
import argparse
import logging

import torch
import torchtext

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
parser.add_argument('--test_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--predict', action='store_true')


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

#################################################################################
# Evaluate model on test set

evaluator = Evaluator(loss=loss, batch_size=opt.batch_size)
loss, accuracy, seq_accuracy = evaluator.evaluate(seq2seq, test)

print("Loss: %f, Word accuracy: %f, Sequence accuracy: %f" % (loss, accuracy, seq_accuracy))


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


    f = open(path,"r")
    lines = f.readlines()
    f.close()

    predictor = Predictor(seq2seq, input_vocab, output_vocab)

    for line in lines:
        input, exp = line.split("\t")
        output = predictor.predict(input.split())
        #print("\n", input)
        print("Output:", " ".join(output), "\nExpected:", exp)

