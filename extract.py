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

def train(model, data, criterion,optimizer, out_dir,batch_size=32,num_epoch=6):
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
        accuracy_total = 0
        evals = 0

        # consuming seen batches from previous training
        for _ in range((epoch - 1) * steps_per_epoch, step):
            next(batch_generator)

        model.train(True)
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
            # if step % 10 == 0:
                # print(output.encode('latin-1', 'replace'))

            # Get Loss
            targets_flattened = target_variables.contiguous().view(-1).long()
            outputs_flattened = output.view(targets_flattened.size(0), -1)
            loss = criterion(outputs_flattened, targets_flattened)

            #
            # # Backward propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Record average loss
            epoch_loss_total += loss.data.item()[0]

            # Compute softmax
            _, predicted = torch.max(outputs_flattened.data, 1)

            # Remove padded targets
            indices = targets_flattened.ne(-1)
            targets_flattened = targets_flattened[indices]
            predicted = Variable(predicted)
            predicted = predicted[indices]

            # Compute Statistics
            accuracy_total += (predicted == targets_flattened).long().sum().data[0]
            evals += len(targets_flattened)

        accuracy = accuracy_total / evals

        epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step)
        print("Total loss:", epoch_loss_total, ", Average Loss:",epoch_loss_avg, ", epoch:", epoch, "accuracy", accuracy)
        epoch_loss_total = 0

    # store initial model to be sure at least one model is stored
    eval_data = data
    loss, accuracy = test(data=eval_data,model=model, criterion=criterion)
    model_name = 'acc_%.2f_ppl_%.2f_s%d' % (accuracy, loss, epoch)
    #best_checkpoints[0] = model_name

    Checkpoint(model=model,
               optimizer=optimizer,
               epoch=epoch, step=step,
               input_vocab=data.fields["src"].vocab,
               output_vocab={}).save(out_dir, name=model_name)

    return model

def test(data, model, criterion, batch_size=32, wrong=None):
    accuracy_total = 0
    loss_total = 0
    evals = 0

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
        # step_elapsed += 1

        input_variables, input_lengths = getattr(batch, 'src')
        target_variables = getattr(batch, 'msk')

        # Forward propagation through classifiers
        output = model(input_variables, input_lengths.tolist(), target_variables)
 #       if(wrong):

        # Get Loss
        targets_flattened = target_variables.contiguous().view(-1).long()
        outputs_flattened = output.view(targets_flattened.size(0), -1)
        loss = criterion(outputs_flattened, targets_flattened)

        # Compute softmax
        _,predicted = torch.max(outputs_flattened.data, 1)

        # Remove padded targets
        indices = targets_flattened.ne(-1)
        targets_flattened = targets_flattened[indices]
        predicted = Variable(predicted)
        predicted = predicted[indices]

        # Compute Statistics
        accuracy_total += (predicted==targets_flattened).long().sum().data[0]
        loss_total += loss.data[0]
        evals += len(targets_flattened)

    accuracy = accuracy_total/evals
    loss = loss_total/step
    return loss, accuracy


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
parser.add_argument('--epochs', type=int, help='Number of epochs', default=6)
parser.add_argument('--num_class', type=int, help='Number of classes', default=2)
parser.add_argument('--weight_vec', help='Weight vector', default=[.5,.5])
parser.add_argument('--print_wrong', type=int, help="provide how often to print",default=100)

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

DC = DiagnosticClassifier(seq2seq, numclass=opt.num_class)
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

all_data = torchtext.data.TabularDataset(
    path=opt.mask_data, format='tsv',
    fields=[('src', src), ('msk', msk)],
    filter_pred=len_filter
)

#TODO: split in train and test?
#train_data,test_data = all_data.split(split_ratio=0.7)
train_data = all_data
test_data = all_data

###########################################################################
# Train Classifier
#TODO: check which hard-coded things should be arguments (see trainer)


# Prepare loss
weight_vec = np.fromstring(opt.weight_vec,sep=",")
loss = CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor(weight_vec))
optimizer = optim.Adam(DC.classifier.parameters(),lr=0.001)

if torch.cuda.is_available():
    loss.cuda()
    #optimizer.cuda()

# Train the classifier
train(data=train_data, model=DC, criterion=loss, optimizer=optimizer, out_dir=opt.output_dir,batch_size=32,num_epoch=opt.epochs)
# # arguments to potentially add
# , expt_dir=opt.output_dir, dev_data=dev, teacher_forcing_ratio=.2, resume=False, checkpoint_path=None)


#################################################################################
# Evaluate model on train set

#TODO: read in the testset
# for now use data which is filled with train-data

loss, accuracy = test(data=test_data, model=DC, criterion=loss,batch_size=32,wrong=opt.print_wrong)

print("\nLoss: %f, Accuracy: %f" % (loss, accuracy))
