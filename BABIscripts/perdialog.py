from seq2seq.dataset import SourceField, TargetField
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.evaluator import Predictor, Evaluator

import torchtext
import torch
import os
import logging

#PATH FOR NORMAL BABI
path = "/home/sanne/machine/data/CLEANED-BABI/babi-dialog"
#PATH FOR BABI+
#path = "/home/sanne/machine/data/CLEANED-BABI/babi+dialog"
trainclean = "task1-trn.txt"
testclean = "task1-tst.txt"

cleanpath = os.path.join(path, trainclean)

src = SourceField()
tgt = TargetField()

max_len = 50
def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

f = open(cleanpath,'r')
lines = f.readlines()
print(lines[0])

train = torchtext.data.TabularDataset(
    path=cleanpath, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
src.build_vocab(train, max_size=50000)
tgt.build_vocab(train, max_size=50000)

###############################################################
###                                                        ###
###                     Training phase                     ###
###                                                        ###
##############################################################

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

for param in model.parameters():
    param.data.uniform_(-0.08, 0.08)

#Prepare loss for training
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)

# Start Training
t = SupervisedTrainer(loss=loss)


print("Start Training: Dialog")

cpath = "/home/sanne/Downloads/"
checkpoint = "machine-checkpoints/"
checkpoint_path = os.path.join(cpath, checkpoint)

model = t.train(model, train,num_epochs=10, checkpoint_path=checkpoint_path) #, epochs=10)

print("\nTraining Accuracy:")
evaluator = Evaluator(loss=loss, batch_size=32)
loss, accuracy = evaluator.evaluate(model, train)
print("Loss: %f, accuracy: %f \n" % (loss, accuracy))


# TODO: This whole thing runs way to fecking quick

##############################################################
###                                                        ###
###                     Testing phase                      ###
###                                                        ###
##############################################################


test_path = os.path.join(path,testclean)
src = SourceField()
tgt = TargetField()

test = torchtext.data.TabularDataset(
    path=test_path, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
src.build_vocab(test, max_size=50000)
tgt.build_vocab(test, max_size=50000)


weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)

print("Start Testing")
evaluator = Evaluator(loss=loss, batch_size=32)
loss, accuracy = evaluator.evaluate(model, test)

print("Loss: %f, accuracy: %f" % (loss, accuracy))
