import logging
import torch
import torchtext

from torch.autograd import Variable
import numpy as np

class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]


class MaskField(torchtext.data.RawField):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True. """
    #TODO: extend this class with required changes to adept to 1 and 0 data
    tensor_types = {
        torch.FloatTensor: float,
        torch.cuda.FloatTensor: float,
        torch.DoubleTensor: float,
        torch.cuda.DoubleTensor: float,
        torch.HalfTensor: float,
        torch.cuda.HalfTensor: float,

        torch.ByteTensor: int,
        torch.cuda.ByteTensor: int,
        torch.CharTensor: int,
        torch.cuda.CharTensor: int,
        torch.ShortTensor: int,
        torch.cuda.ShortTensor: int,
        torch.IntTensor: int,
        torch.cuda.IntTensor: int,
        torch.LongTensor: int,
        torch.cuda.LongTensor: int
    }

    def __init__(self, sequential=True, #use_vocab=True, init_token=None,eos_token=None, fix_length=None,
                 tensor_type=torch.FloatTensor, preprocessing=None, postprocessing=None, #lower=False,
                 #tokenize=(lambda s: s.split()), include_lengths=False,
                 batch_first=False, pad_token=-1, unk_token="<unk>",pad_first=False, truncate_first=False
                 ):
        self.sequential = sequential
#        self.use_vocab = use_vocab
#        self.init_token = init_token
#        self.eos_token = eos_token
#        self.unk_token = unk_token
#        self.fix_length = fix_length
        self.tensor_type = tensor_type
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
#        self.lower = lower
#        self.tokenize = get_tokenizer(tokenize)
#        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
#        self.pad_first = pad_first
#        self.truncate_first = truncate_first


    def process(self, batch, *args, **kargs):
        device = None if torch.cuda.is_available() else -1
        if self.sequential:
            batch2 = []
            for item in batch:
                #print(item)
                batch2.append(np.fromstring(item, dtype=float, sep=' '))
            batch = batch2
        else:
            batch = np.fromstring(batch, dtype=float, sep=' ')
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)

        if self.tensor_type not in self.tensor_types:
            raise ValueError(
                "Specified Field tensor_type {} can not be used with ".format(self.tensor_type))

        #TODO: Introduce padding - intuitively this breaks, what to padd with?? -1?
        padded = self.pad(batch)
        batch = self.tensor_type(padded)
        if not device:
            batch = batch.cuda() # Potentially requires .cuda(device)
        batch = Variable(batch)
        #print(batch)
        return batch

    def pad(self, batch):
        batch = list(batch)
        #print("test1",batch)
        if not self.sequential:
            return batch
        max_len = max(len(x) for x in batch)
        #TODO: figure out if the enters introduced below are a problem
        padded, lengths = [], []
        for x in batch:
            padded.append(
                #list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                np.append(x[:max_len],
                [self.pad_token] * max(0, max_len - len(x))
            ))
        #     lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        # if self.include_lengths:
        #     return (padded, lengths)
        return padded
