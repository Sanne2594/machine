import torch.nn as nn
import torch.nn.functional as F

class DiagnosticClassifier(nn.Module):

    def __init__(self, original_model, hidden_encoder_dim, type=None):
        super(DiagnosticClassifier, self).__init__()
        self.encoder = original_model.encoder
        #TODO: freeze weigths

        if type=="binary":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_encoder_dim, 64), # Asumes hidden_encoder_dim to be (bigger than) 128
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid() #Forces the model to stay close to either 0 or 1.
            )
        else:
            print("No Classifier type provided, expected binary or ...")

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        result = self.classifier(encoder_hidden)
        return result