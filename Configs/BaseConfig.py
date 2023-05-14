import torch.nn.functional as F
from Data import Vocab, MyDataset

class BaseConfig():
    #TRANSFORMER
    def __init__(self):
        #==============DATA=================
        
        ##VOCAB 
        self.DATA_VOCAB_name = "default"

        ##DATASET
        self.DATA_DATASET_train = ""
        #self.DATA.DATASET.vad = ""
        #self.DATA.DATASET.test = ""
        self.DATA_DATASET_truncate_src = True
        self.DATA_DATASET_max_src_len = 510

        self.DATA_DATASET_truncate_tgt = False
        self.DATA_DATASET_max_tgt_len = None
        self.DATA_DATASET_max_rows = 100
        self.DATA_DATASET_min_freq = 0

        #==============ENGINE=================
        
        #Dataloader
        self.ENGINE_TRAINER_batch_size = 10
        self.ENGINE_TRAINER_shuffle = True
        
        #Loss
                
        #Optim
        ###Adam
        self.ENGINE_TRAINER_betas = (0.9, 0.98)
        self.ENGINE_TRAINER_eps = 1e-09
        self.ENGINE_TRAINER_lr = 0.01

        # Trainer
        self.ENGINE_TRAINER_checkpoint = "Model/models/model.pth"  # to save params
        self.ENGINE_TRAINER_epochs = 5
        self.ENGINE_TRAINER_pretrain = ""                        # to load params

        #==============MODEL=================        

        self.d_model = 512
        self.num_heads = 8
        self.num_encoder_layers=1
        self.num_decoder_layers = 1
        self.forward_expansion = 4
        self.dropout = 0.1
        self.max_len = 510
        self.device = "cuda"

        