import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import torchtext
from torch.autograd import Variable

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models import DiagnosticClassifier
from seq2seq.trainer import SupervisedTrainer


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--output_dir', default='../models', help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
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

##################################################################################
# create diagnostic classifier

#TODO: figure out how to extract hidden layer size from model
encoder_hidden_dim = 128
DC = DiagnosticClassifier(seq2seq, encoder_hidden_dim, type="binary")
if torch.cuda.is_available():
    DC.cuda()

#TODO: should we do this for parameters in classifier?
# for param in DC.parameters():
#     param.data.uniform_(-0.08, 0.08)



############################################################################
# Prepare dataset and masks
src = SourceField()
#TODO: create datatype MaskField??
msk = TargetField()
src.vocab = input_vocab
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len

data = torchtext.data.TabularDataset(
    path=opt.mask_data, format='tsv',
    fields=[('src', src), ('msk', msk)],
    filter_pred=len_filter
)

#TODO: split in train and test?

###########################################################################
# Train Classifier
#TODO: check which hard-coded things should be arguments (see trainer)

# Prepare loss
loss = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss.cuda()

# create trainer
t = SupervisedTrainer(loss=loss, batch_size=32,
                      checkpoint_every=100,
                      print_every=100, expt_dir=opt.output_dir)

#TODO: check is this routine is apropriate. It isn't
DC = t.train(DC, data,
                  num_epochs=6, #dev_data=dev,
                  optimizer='adam',
                  teacher_forcing_ratio=.2,
                  learning_rate=0.001,
                  resume=False,
                  checkpoint_path=None)


#################################################################################
# Evaluate model on train set

evaluator = Evaluator(loss=loss, batch_size=32)
loss, accuracy, seq_accuracy = evaluator.evaluate(DC, data)

print("\nLoss: %f, Word accuracy: %f, Sequence accuracy: %f" % (loss, accuracy, seq_accuracy))
