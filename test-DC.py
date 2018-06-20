import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import torchtext
from torch.autograd import Variable

import seq2seq
from seq2seq.dataset import SourceField, MaskField
from seq2seq.evaluator import Evaluator, Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.models import DiagnosticClassifier
from seq2seq.trainer import SupervisedTrainer
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def test(data, model, criterion, batch_size=32, wrong=None):
    accuracy_total = 0
    loss_total = 0
    evals_tot = 0

    device = None if torch.cuda.is_available() else -1
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)

    batch_generator = batch_iterator.__iter__()
    step = 0
    for batch in batch_generator:
        step = step + 1
        input_variables, input_lengths = getattr(batch, 'src')
        target_variables = getattr(batch, 'msk')
        output = model(input_variables, input_lengths.tolist(), target_variables)

        accuracy = 0
        loss=0
        evals = 0
        for i in range(batch_size):
            loss += criterion(output[i],target_variables[i].long())
            predicted = torch.max(output[i],1)
            indices = target_variables[i].ne(-1)
            targets = target_variables[i][indices].long()
            predicted = predicted[1][indices]
            inputs = input_variables[i][indices]
            thresh = (predicted==targets).long().sum().data[0]/len(targets)
            accuracy += (predicted==targets).long().sum().data[0]
            evals += len(targets)

            if thresh <= wrong:
                inputs = [input_vocab.itos[int(tok.data)] for tok in inputs]
                print("Input:", inputs)
                print("Output:", ' '.join(['%i' % i for i in predicted]))
                print("Target:", ' '.join(['%i' % i for i in targets]))
                print("Accuracy:", thresh)
                input()
        accuracy_total += accuracy
        loss_total += loss.data[0]
        evals_tot += evals
    accuracy = accuracy_total/evals_tot
    loss = loss_total/step
    return loss, accuracy

def add_stats(data,model,batch_size=32):
    true_pos=0
    out_pos = 0
    tar_pos= 0

    device = None if torch.cuda.is_available() else -1
    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)

    batch_generator = batch_iterator.__iter__()
    step = 0
    for batch in batch_generator:
        step = step + 1
        input_variables, input_lengths = getattr(batch, 'src')
        target_variables = getattr(batch, 'msk')
        output = model(input_variables, input_lengths.tolist(), target_variables)
        targets_flattened = target_variables.contiguous().view(-1).long()
        outputs_flattened = output.view(targets_flattened.size(0), -1)
        _,predicted = torch.max(outputs_flattened.data, 1)

        # Remove padded targets
        indices = targets_flattened.ne(-1)
        targets_flattened = targets_flattened[indices]
        predicted = Variable(predicted)
        predicted = predicted[indices]

        tar_ind = [i for i,val in enumerate(targets_flattened) if (val>0).data.numpy()]
        out_ind = [i for i,val in enumerate(predicted) if (val>0).data.numpy()]
        true_ind = [i for i in tar_ind if i in out_ind]

        true_pos += len(true_ind)
        tar_pos += len(tar_ind)
        out_pos += len(out_ind)

        # true_pos += (predicted==1 and targets_flattened==1).long().sum().data[0]
        # out_pos += (predicted==1).long().sum().data[0]
        # tar_pos += (targets_flattened==1).long().sum().data[0]

    precision = true_pos/(out_pos)
    recall = true_pos/(tar_pos)
    return precision,recall

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--mask_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--weight_vec', help='Weight vector', default=[.5,.5])
parser.add_argument('--print_wrong', type=float, help="decide below which value to print",default=.8)
parser.add_argument('--stats', action='store_true')

opt = parser.parse_args()

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

#logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
DC = checkpoint.model
input_vocab = checkpoint.input_vocab
#output_vocab = checkpoint.output_vocab


############################################################################
# Prepare dataset and masks
src = SourceField()
msk = MaskField()
src.vocab = input_vocab
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len

all_data = torchtext.data.TabularDataset(
    path=opt.mask_data, format='tsv',
    fields=[('src', src), ('msk', msk)],
    filter_pred=len_filter
)

#################################################################################
# Evaluate model on train set

weight_vec = np.fromstring(opt.weight_vec,sep=",")
loss = CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor(weight_vec))

if torch.cuda.is_available():
    loss.cuda()
    #optimizer.cuda()


#TODO: read in the testset
# for now use data which is filled with train-data

if opt.stats:
    precision,recall = add_stats(data=all_data,model=DC,batch_size=32)
    print("\nPrecision: %f, Recall: %f" %(precision,recall))
else:
    loss, accuracy = test(data=all_data, model=DC, criterion=loss,batch_size=opt.batch_size,wrong=opt.print_wrong)
    print("\nLoss: %f, Accuracy: %f" % (loss, accuracy))
