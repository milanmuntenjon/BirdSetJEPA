import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
import hydra.utils
from omegaconf import DictConfig

class IJEPAPretrainModule(L.LightningModule):
    """
    Pretraining module for I-JEPA with BirdSet-style integration.
    Works with IJEPABackbone and ignores labels from the datamodule.
    """

    def __init__(
        self,
        network,                 
        lr: float = 1e-4,
        weight_decay: float = 0.02,
        loss_fn= None,
        
        **args,
    ):
        super().__init__()
        self.model = network.model
        self.loss_fn = network.model.loss_fn
        self.lr = lr
        self.weight_decay = weight_decay


    def training_step(self, batch, batch_idx):
        x = batch["input_values"]


        z, h = self.model(x)      
        loss = self.loss_fn(z, h)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        

        self.model.update_teacher()

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_values"]

        z, h = self.model(x)
        loss = self.model.loss_fn(z, h)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):

        return AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
