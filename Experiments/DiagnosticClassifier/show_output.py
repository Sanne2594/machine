import os
import argparse
import torch
from seq2seq.util.checkpoint import Checkpoint
import numpy as np
import torchtext
from torch.autograd import Variable

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, MaskField
from seq2seq.evaluator import Evaluator, Predictor
import sys

model_loc = "Model-DC-ET/acc_0.90_ppl_0.08_s100"
data_loc = "DataGeneration/disfluency-masks/train_alteration-masks.txt"

# print(sys.path)
# model_loc = "API-calls/results/Model-DC-cuis-1/acc_0.46_ppl_1.30_s1000"
# data_loc = "API-calls/Data-1/cuisine_masks.txt"

# Read in model
checkpoint = Checkpoint.load(model_loc)
model = checkpoint.model
input_vocab = checkpoint.input_vocab

# Read in data
src = SourceField()
msk = MaskField()
src.vocab = input_vocab
max_len = 75

def len_filter(example):
    return len(example.src) <= max_len

data = torchtext.data.TabularDataset(
    path=data_loc, format='tsv',
    fields=[('src', src), ('msk', msk)],
    filter_pred=len_filter
)

device = None if torch.cuda.is_available() else -1
batch_iterator = torchtext.data.BucketIterator(
    dataset=data, batch_size=1,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device, repeat=False)

batch_generator = batch_iterator.__iter__()

# Do predictions.
for batch in batch_generator:
    input_variables, input_lengths = getattr(batch, 'src')
    target_variables = getattr(batch, 'msk')

    output = model(input_variables, input_lengths.tolist(), target_variables)

    _, predicted = torch.max(output.data, 2)

    # for vertical in input_variables:
    #     input = [input_vocab.itos[int(tok.data)] for tok in vertical]

    # inputs = [input_vocab.itos[int(tok.data)] for tok in input_variables[0]]
    # print("Input:",inputs)
    # print("Output:",' '.join(['%i' % i for i in predicted[0][-10:]]))
    # print("Target:",' '.join(['%i' % i for i in target_variables[0][-10:]]))
    #
    # Remove padded targets
    indices = target_variables[0].ne(-1)
    targets_flattened = target_variables[0][indices]
    predicted_new = Variable(predicted)
    predicted_new = predicted_new[indices]

    # Compute Statistics
    accuracy= (predicted_new == targets_flattened.long()).long().sum().data[0]/len(targets_flattened)
    #loss_value = loss.data[0]
    if accuracy != 0:
        if sum(predicted[0]) >0:
            inputs = [input_vocab.itos[int(tok.data)] for tok in input_variables[0]]
            print("Input:",inputs)
            print("Output:",' '.join(['%i' % i for i in predicted[0]]))
            print("Target:",' '.join(['%i' % i for i in target_variables[0]]))
            input()
    # input()
