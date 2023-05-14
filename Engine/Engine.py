import os
import torch
from tqdm import tqdm

from Model.utils.masking import create_masks


class Trainer():
    def __init__(self, model, criterion, optim, epochs, dataloader, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.epochs = epochs
        self.dataloader = dataloader
        self.device = device
        self.scheduler = scheduler
        
        self.cur_epoch = 0
        self.loss = -1
        
    def train(self, pretrain = ""):
        if os.path.exists(pretrain):
            self.load_state(pretrain)
        
        model = self.model.to(self.device)
        model.train()
        soe = self.cur_epoch
        for epoch in range(self.epochs):
          total_loss = []
          for batch in tqdm(self.dataloader, desc=f"### Epoch {soe + epoch + 1}"): 
              x = torch.LongTensor(batch['x']).to(self.device)
              y = torch.LongTensor(batch['y']).to(self.device)
              mask_x, mask_y = create_masks(torch.LongTensor(batch['x']), torch.LongTensor(batch['y']), self.device)
              
              output = model(x, y, mask_x, mask_y)
              
              vocab_size = output.shape[-1]

              loss = self.criterion(output.reshape(-1, vocab_size), y.reshape(-1))
              total_loss.append(loss.item())
              self.optim.zero_grad()
              loss.backward()
              self.optim.step()
              if self.scheduler is not None:
                    self.scheduler.step()

          self.loss = sum(total_loss) / len(total_loss)
          self.cur_epoch = epoch
          tqdm.write(f"Avg loss: {self.loss}")
    

    
    def __call__(self, pretrain = ""):
        self.train(pretrain)
        
    def save_state(self, save_dir="Model/model/model.pth"):
        torch.save({
            'epoch': self.cur_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': self.loss,
            }, save_dir)
        
    def load_state(self, load_dir=None):
        state = torch.load(load_dir)
        self.cur_epoch = state['epoch']
        self.model.load_state_dict(state['model_state_dict'])
        self.optim.load_state_dict(state['optimizer_state_dict'])
        self.loss = state['loss']

        print(f"Got state from {load_dir}")
