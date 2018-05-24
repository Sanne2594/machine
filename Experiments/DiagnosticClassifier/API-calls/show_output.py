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


model_loc = "results/Models-1/acc_0.46_ppl_1.30_s1000"
data_loc = "Data/cuisine_masks.txt"

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

count = 0

# Do predictions.
for batch in batch_generator:
    count += 1
    input_variables, input_lengths = getattr(batch, 'src')
    target_variables = getattr(batch, 'msk')

    output = model(input_variables, input_lengths.tolist(), target_variables)
    _, predicted = torch.max(output.data, 2)

    for tok in input_variables:
        print(tok)
        input_vocab.itos[int(tok.data)]
    #input = [input_vocab.itos[int(tok.data)] for tok in input_variables]
    print("Input:",input)
    print("Output:",predicted)
    print("Target:",target_variables)

    indices = target_variables.ne(-1)
    targets_flattened = target_variables[indices]
    predicted = Variable(predicted)
    predicted = predicted[indices]

    mistakes = []
    for i, tar in enumerate(targets_flattened):
        input_variables.type("torch.FloatTensor")
        if tar == predicted[i]:
            mistakes.append(input_variables[i])
#    mistakes = [input_variables[i] for i,tar in enumerate(targets_flattened) if tar==predicted[i.type()]]
    print(mistakes)
    if count >= 1:
        dfghj