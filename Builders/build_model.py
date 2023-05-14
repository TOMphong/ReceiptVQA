from .build_data import build_data
from Model import Transformer



def build_model(config, tokenize):
    data, vocab = build_data(config, tokenize)
    src_vocab = vocab
    trg_vocab = vocab
    model = Transformer(config=config,
                        src_vocab=src_vocab,
                        trg_vocab=trg_vocab)
    
    return model, data