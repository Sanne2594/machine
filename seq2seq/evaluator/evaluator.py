from __future__ import print_function, division

import torch
import torchtext
from collections import defaultdict

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data, threshold=0., select_eval=False):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
            threshold (float): optional threshold of accuracy under which to print output

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()

        word_match = 0
        word_total = 0

        seq_match = 0
        seq_total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        src_vocab = data.fields[seq2seq.src_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]
        unk = tgt_vocab.stoi['<unk>']
        api = tgt_vocab.stoi['api_call']
        eos = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].SYM_EOS]

        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']

            match_per_seq = torch.zeros(batch.batch_size).type(torch.FloatTensor)
            total_per_seq = torch.zeros(batch.batch_size).type(torch.FloatTensor)

            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                non_api = target.ne(api)
                non_eos = target.ne(eos)
                non_unk = target.ne(unk)

                if select_eval:
                    consider = non_padding*non_api*non_eos*non_unk
                else:
                    consider = non_padding

                correct_per_seq = seqlist[step].view(-1).eq(target).data * consider.data

                # print("correct per seq: ", correct_per_seq)
                # raw_input()

                match_per_seq += correct_per_seq.type(torch.FloatTensor)
                total_per_seq += consider.data.type(torch.FloatTensor)

            word_match += match_per_seq.sum()
            word_total += total_per_seq.sum()

            # print('match per seq', match_per_seq)
            # print('total per seq', total_per_seq)
            # raw_input()

            seq_match += match_per_seq.eq(total_per_seq).sum()
            seq_total += total_per_seq.shape[0]

            #Compute accuracy per sequence
            if not seq_match ==0:
                curr_acc = (match_per_seq.sum() / total_per_seq.sum())
                if curr_acc < threshold:
                    if torch.cuda.is_available():
                        tgt_seq = [tgt_vocab.itos[tok] for tok in target_variables.data.cpu().numpy()[0]]
                        in_seq = [src_vocab.itos[tok] for tok in input_variables.data.cpu().numpy()[0]]
                    else:
                        tgt_seq = [tgt_vocab.itos[tok] for tok in target_variables.data.numpy()[0]]
                        in_seq = [src_vocab.itos[tok] for tok in input_variables.data.numpy()[0]]
                    print("\nModel input:", in_seq)
                    length = other['length'][0]
                    out_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
                    out_seq = [tgt_vocab.itos[tok] for tok in out_id_seq]
                    print("Model output:", out_seq)
                    print("Target output:", tgt_seq)
                    print("Word accuracy:", match_per_seq.sum() / total_per_seq.sum())

        if word_total == 0:
            accuracy = float('nan')
        else:
            accuracy = word_match / word_total

        if seq_total == 0:
            seq_accuracy = float('nan')
        else:
            seq_accuracy = seq_match/seq_total

        return loss.get_loss(), accuracy, seq_accuracy

    def get_cooccurence_matrix(self, model, data):
        """ Compute cooccurences for output and input words given by 
            the attention generated by the model.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        model.eval()

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        att_dict = defaultdict(lambda: defaultdict(float))
    
        # loop over batches
        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Output sequences and attentions
            seqlist = other['sequence']
            attentions = [att.squeeze() for att in other['attention_score']]

            # loop over input variables in batch, get predictions
            for q, input_seq in enumerate(input_variables.data):
                prediction = [step.squeeze().data[q] for step in seqlist]
                for i, output_word in enumerate(prediction):
                    if target_variables.data[q][i+1] != pad:
                        # put correct target in attention dictionary
                        for j, input_word in enumerate(input_seq):
                            att = attentions[i][q][j].data[0]
                            att_dict[output_word][input_word] += att


                        # if output_word in att_dict:
                        #     output_word_dict = att_dict[output_word]
                        # else:
                        #     output_word_dict = {}
                        #     att_dict[output_word] = output_word_dict
                        # for j, input_word in enumerate(input_seq):
                        #     if input_word in output_word_dict:
                        #         output_word_dict[input_word].append(attentions[i][q][j].data.cpu().numpy()[0])
                        #     else:
                        #         output_word_dict[input_word] = [attentions[i][q][j].data.cpu().numpy()[0]]

        return att_dict
