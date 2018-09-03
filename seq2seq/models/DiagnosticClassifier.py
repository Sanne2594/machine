import torch.nn as nn
import torch.nn.functional as F

class DiagnosticClassifier(nn.Module):

    def __init__(self, original_model, numclass=2):
        super(DiagnosticClassifier, self).__init__()
        self.encoder = original_model.encoder
        # Freeze weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_encoder_dim = self.encoder.hidden_size
        inner = int(hidden_encoder_dim+numclass/2)
        self.classifier = nn.Sequential(
            #nn.Linear(hidden_encoder_dim, numclass),
            # Use this code when results appear to not regressinformation properly
            nn.Linear(hidden_encoder_dim, inner),
            nn.ReLU(),
            nn.Linear(inner, numclass),
            nn.LogSoftmax()
            )

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        result = self.classifier(encoder_outputs)
        return result

    def flatten_parameters(self):
        print("Does nothing")