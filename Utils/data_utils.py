from torch.nn import functional as F
import torch
import numpy as np
from Data.vocab import Vocab

def padding(vector: list, max_len):
  pad_dim = (0, max_len - len(vector))
  return F.pad(vector, pad_dim, 'constant').tolist()

def my_collate(batch):
    max_x = np.max([item['x_len'] for item in batch])
    max_y = np.max([item['y_len'] for item in batch])

    x = [ F.pad(torch.Tensor([Vocab.BOS] + item['x'] + [Vocab.EOS]), (0, max_x-len(item['x'])),'constant').tolist() for item in batch ]  ## (1,max_x) list
    y = [ F.pad(torch.Tensor([Vocab.BOS] + item['y'] + [Vocab.EOS]), (0, max_y-len(item['y'])),'constant').tolist() for item in batch ]  ## (1,max_y) list

    src = [item['src'] for item in batch]
    tgt = [item['tgt'] for item in batch]

    return {'x': x,
            'y': y,
            'src': src,
            'tgt': tgt,
            'x_len': [item['x_len']+2 for item in batch],
            'y_len': [item['y_len']+2 for item in batch]
            }
