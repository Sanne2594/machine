/home/sanne/.local/lib/python3.5/site-packages/torch/serialization.py:325: SourceChangeWarning: source code of class 'torch.nn.modules.rnn.LSTM' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/home/sanne/.local/lib/python3.5/site-packages/torch/serialization.py:325: SourceChangeWarning: source code of class 'torch.nn.modules.sparse.Embedding' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/home/sanne/.local/lib/python3.5/site-packages/torch/serialization.py:325: SourceChangeWarning: source code of class 'torch.nn.modules.linear.Linear' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
Traceback (most recent call last):
  File "extract.py", line 162, in <module>
    DC = DiagnosticClassifier(seq2seq, numclass=num_class)
  File "/home/sanne/machine/sanne/machine/seq2seq/models/DiagnosticClassifier.py", line 18, in __init__
    nn.Linear(hidden_encoder_dim, inner),
  File "/home/sanne/.local/lib/python3.5/site-packages/torch/nn/modules/linear.py", line 41, in __init__
    self.weight = Parameter(torch.Tensor(out_features, in_features))
TypeError: torch.FloatTensor constructor received an invalid combination of arguments - got (float, int), but expected one of:
 * no arguments
 * (int ...)
      didn't match because some of the arguments have invalid types: (!float!, !int!)
 * (torch.FloatTensor viewed_tensor)
 * (torch.Size size)
 * (torch.FloatStorage data)
 * (Sequence data)

