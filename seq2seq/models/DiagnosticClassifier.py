import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):

    #TODO: introduce a variable that keeps track of which type of model is build
    def __init__(self, original_model, output_info):
        super(Seq2seq, self).__init__()
        self.encoder = original_model.encoder
        #TODO: freeze weigths

        #TODO: figure this out: kijken in de git van ivd?
        self.decoder = nn.Linear(n_input, n_hidden)

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        #TODO: Zelfde als hierboven
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result