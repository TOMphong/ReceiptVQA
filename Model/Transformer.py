import torch
import math
from torch import nn, Tensor



class Transformer(nn.Module):
    def __init__(
        self,
        config,
        src_vocab,
        trg_vocab
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(len(src_vocab), config.d_model)
        self.src_position_embedding = nn.Embedding(src_vocab.max_len+2, config.d_model)
        self.trg_word_embedding = nn.Embedding(len(trg_vocab), config.d_model)
        self.trg_position_embedding = nn.Embedding(trg_vocab.max_len+2, config.d_model)

        self.device = config.device
        self.transformer = nn.Transformer(
            config.d_model,
            config.num_heads,
            config.d_model,
            config.d_model,
            config.forward_expansion,
            config.dropout,
        )
        self.fc_out = nn.Linear(config.d_model, len(trg_vocab))
        self.dropout = nn.Dropout(config.dropout)
        self.src_pad_idx = src_vocab.PAD

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):

        N, src_seq_length = src.shape
        N, trg_seq_length = trg.shape

        src = src.view(src_seq_length, N)
        trg = trg.view(trg_seq_length, N)

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )
        

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out