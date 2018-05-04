import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import torchtext
from torch.autograd import Variable

import seq2seq
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.dataset import SourceField, MaskField
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models import DiagnosticClassifier
from seq2seq.trainer import SupervisedTrainer
import torch.optim as optim


def train(model, data, criterion,optimizer, batch_size=32,num_epoch=6):
    epoch_loss_total = 0  # Reset every epoch

    device = None if torch.cuda.is_available() else -1

    #TODO: hier gaat het fout
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)

    steps_per_epoch = len(batch_iterator) #was 10, when printed
    total_steps = steps_per_epoch * num_epoch
    step = 0

    for epoch in range(1, num_epoch + 1):
        batch_generator = batch_iterator.__iter__()
        #torchtext.data.iterator.BucketIterator

        # consuming seen batches from previous training
        for _ in range((epoch - 1) * steps_per_epoch, step):
            next(batch_generator)

        model.train(True)
        print(batch_generator)
        #<generator object Iterator.__iter__ at 0x7fbc17680e60>
        for batch in batch_generator:
            #TODO: does not reach this place
            step = step+1
            #step_elapsed += 1

            input_variables, input_lengths = getattr(batch, 'src')
            target_variables = getattr(batch, 'msk')

            # Forward propagation through classifiers
            output = model(input_variables, input_lengths.tolist(), target_variables)
            # Possibly add teacher forcing ratio here
            #         32 x     <75     x 2
            # batch size * max lengths * num_class

            # Get loss
            loss = criterion(output,target_variables)

            # Backward propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Record average loss
            epoch_loss_total += loss.get_loss()
            print(step)
        epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step)
        print("Total loss:", epoch_loss_total, ", Average Loss:",epoch_loss_avg, ", epoch:", epoch)
        epoch_loss_total = 0

    return model


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

#logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

##################################################################################
# create diagnostic classifier

#TODO: figure out how to extract hidden layer size from model
encoder_hidden_dim = 128
num_class=2
DC = DiagnosticClassifier(seq2seq, numclass=num_class)
if torch.cuda.is_available():
    DC.cuda()

#TODO: should we do this for parameters in classifier?
# for param in DC.parameters():
#     param.data.uniform_(-0.08, 0.08)



############################################################################
# Prepare dataset and masks
src = SourceField()
msk = MaskField()
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
loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
#forward(self, input, target)
optimizer = optim.Adam(DC.classifier.parameters(),lr=0.001)

if torch.cuda.is_available():
    loss.cuda()
    optimizer.cuda()

num_epoch = 6
# Train the classifier
train(data=data, model=DC, criterion=loss, optimizer=optimizer, batch_size=32,num_epoch=6)

# # arguments to potentially add
# , expt_dir=opt.output_dir, dev_data=dev, teacher_forcing_ratio=.2, resume=False, checkpoint_path=None)


#################################################################################
# Evaluate model on train set

#TODO: make this compatable with cross entropy loss
evaluator = Evaluator(loss=loss, batch_size=32)
loss, accuracy, seq_accuracy = evaluator.evaluate(DC, data)

print("\nLoss: %f, Word accuracy: %f, Sequence accuracy: %f" % (loss, accuracy, seq_accuracy))
