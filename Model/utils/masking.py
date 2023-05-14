import torch
from torch import Tensor
from Data.vocab import Vocab

def create_masks(src_batch: Tensor, tgt_batch: Tensor, device: str = "cpu"):
    # ----------------------------------------------------------------------
    # [1] padding mask
    # ----------------------------------------------------------------------
    
    # (batch_size, 1, max_tgt_seq_len)
    src_pad_mask = (src_batch != Vocab.PAD).unsqueeze(1)
    
    # (batch_size, 1, max_src_seq_len)
    tgt_pad_mask = (tgt_batch != Vocab.PAD).unsqueeze(1)

    # ----------------------------------------------------------------------
    # [2] subsequent mask for decoder inputs
    # ----------------------------------------------------------------------
    max_tgt_sequence_length = tgt_batch.shape[1]
    tgt_attention_square = (max_tgt_sequence_length, max_tgt_sequence_length)

    # full attention
    full_mask = torch.full(tgt_attention_square, 1)
    
    # subsequent sequence should be invisible to each token position
    subsequent_mask = torch.tril(full_mask)
    
    # add a batch dim (1, max_tgt_seq_len, max_tgt_seq_len)
    subsequent_mask = subsequent_mask.unsqueeze(0)

    return src_pad_mask.to(device), (tgt_pad_mask & subsequent_mask).to(device)

