from Data import *


def build_data(config, tokenize):
    data = MyDataset(config=config,
                    tokenize = tokenize
                    )
    
    return data, data.vocab

