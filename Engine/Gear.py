from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
from torch import Tensor

def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


class Scheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int = 4000,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super(Scheduler, self).__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups



class BaseLoss(nn.Module):
    def __init__(self, label_smoothing: float=0.0) -> None:
        super(BaseLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(ignore_index  = 0,
                                             label_smoothing = label_smoothing)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        return self.loss_func(logits, labels)