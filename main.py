from Utils import tokenizing
from Builders.build_trainer import build_trainer
from Utils import my_collate

from Configs.BaseConfig import BaseConfig

if __name__ == "__main__":
    config = BaseConfig()

    trainer, pretrain = build_trainer(config, tokenizing, my_collate)
    
    trainer.train(pretrain=pretrain)

    
    