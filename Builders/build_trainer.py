from .build_model import build_model
from Engine import Trainer, Scheduler, BaseLoss

from torch.utils.data import DataLoader
import torch
from torch import nn




def build_trainer(config, tokenize, collate_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = build_model(config,
                              tokenize)
    traindata = DataLoader(dataset = data, 
                           batch_size = config.ENGINE_TRAINER_batch_size, 
                           collate_fn = collate_fn, 
                           shuffle = config.ENGINE_TRAINER_shuffle)
    criterion =  BaseLoss()
    
    optimizer = torch.optim.Adam(params = model.parameters(), 
                                 betas = config.ENGINE_TRAINER_betas, 
                                 eps = config.ENGINE_TRAINER_eps, 
                                 lr = config.ENGINE_TRAINER_lr)

    scheduler = Scheduler(optimizer=optimizer,
                          dim_embed=config.d_model)

    trainer = Trainer(model = model,
                     criterion = criterion,
                     optim = optimizer,
                     epochs = config.ENGINE_TRAINER_epochs,
                     dataloader = traindata,
                     scheduler = scheduler,
                     device = device)
    
    pretrain = config.ENGINE_TRAINER_pretrain

    return trainer, pretrain
