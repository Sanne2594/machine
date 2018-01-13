import os
path = "/home/sanne/machine/data/CLEANED-SCAN/simple_split/"
import torch.nn as nn
from seq2seq.util.checkpoint import Checkpoint

from seq2seq.dataset import SourceField, TargetField
import torchtext

import torch
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity

from seq2seq.evaluator import Predictor, Evaluator

### READ CHECKPOINT
# checkpoint = Checkpoint.load(checkpoint_path)
# seq2seq = checkpoint.model

##############################################################
###                                                        ###
###                     Training phase                     ###
###                                                        ###
##############################################################

#Read in train data
src = SourceField()
tgt = TargetField()
max_len = 50

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

filename = "tasks_train_simple.txt"
fpath = os.path.join(path,filename)

train = torchtext.data.TabularDataset(
    path=fpath, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

src.build_vocab(train, max_size=50000)
tgt.build_vocab(train, max_size=50000)
# input_vocab = checkpoint.input_vocab
# output_vocab = checkpoint.output_vocab
# src.vocab = input_vocab
# tgt.vocab = output_vocab

#Initialize model
hidden_size_in = 128
hidden_size_out = 128
sos_id = tgt.sos_id
eos_id = tgt.eos_id

embedding_size = 128
n_epochs = 10
start_epoch = 1
start_step = 1

encoder = EncoderRNN(len(src.vocab), max_len, hidden_size_in,embedding_size,variable_lengths=True)
decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size_out, sos_id, eos_id,)
model = Seq2seq(encoder, decoder)

# TODO: check if makes sense to copy this
for param in model.parameters():
    param.data.uniform_(-0.08, 0.08)

#Prepare loss for training
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)

# Start Training
t = SupervisedTrainer(loss=loss)

""" Run training for a given model.
(voor t.train)
Args:
    model (seq2seq.motraindels): model to run training on, if `resume=True`, it would be
       overwritten by the model loaded from the latest checkpoint.
    data (seq2seq.dataset.dataset.Dataset): dataset object to train on
    num_epochs (int, optional): number of epochs to run (default 5)
    resume(bool, optional): resume training with the latest checkpoint, (default False)
    dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
    optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
       (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
    teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
    learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
    checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
    top_k (int): how many models should be stored during training
Returns:
    model (seq2seq.models): trained model.
"""
print("Start Training")

cpath = "/home/sanne/Downloads/"
checkpoint = "machine-checkpoints/"
checkpoint_path = os.path.join(cpath, checkpoint)

model = t.train(model, train,checkpoint_path=checkpoint_path) #, epochs=10)

print("Training Accuracy:")
evaluator = Evaluator(loss=loss, batch_size=32)
loss, accuracy = evaluator.evaluate(model, train)
print("Loss: %f, accuracy: %f \n" % (loss, accuracy))

##############################################################
###                                                        ###
###                     Testing phase                      ###
###                                                        ###
##############################################################

#  generate test set
filename = "tasks_test_simple.txt"
fpath = os.path.join(path,filename)

test = torchtext.data.TabularDataset(
    path=fpath, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)

print("Start Testing")
evaluator = Evaluator(loss=loss, batch_size=32)
loss, accuracy = evaluator.evaluate(model, test)

print("Loss: %f, accuracy: %f" % (loss, accuracy))

# Start training at 11:20
# Duurt ongeveer 10 minuten in totaal
#

