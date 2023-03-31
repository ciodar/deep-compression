import pytorch_lightning as lit
import trainer.metrics as module_metric

import torch
import torch.nn.functional as F


class LitModel(lit.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = getattr(F, config['loss'])
        self.metric_ftns = [getattr(module_metric, met) for met in config['metrics']]


    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for met in self.metric_ftns:
            self.log(met.__name__, met(output, target), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # compute metrics
        for met in self.metric_ftns:
            self.log('val_' + met.__name__, met(logits, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # compute metrics
        for met in self.metric_ftns:
            self.log('test_' + met.__name__, met(logits, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        # log model parameters
        if self.logger:
            tensorboard = self.logger.experiment
            for name, p in self.model.state_dict().items():
                tensorboard.add_histogram(name, p, self.global_step)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.config.init_obj('optimizer', torch.optim, trainable_params)
        if 'lr_scheduler' in self.config.config:
            lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
            scheduler_dict = {"scheduler": lr_scheduler, "interval": "epoch"}
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                mnt = self.config['trainer']['monitor'].split()[1]
                scheduler_dict['monitor'] = mnt
        else:
            return optimizer
        return [optimizer], [scheduler_dict]
