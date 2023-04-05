import torch
from abc import abstractmethod
from numpy import inf

import pytorch_lightning as lit

import trainer.callbacks as module_callback


def get_trainer(config):
    cfg_trainer = config['trainer']

    if torch.cuda.is_available():
        accelerator, devices = "auto", config['n_gpu']
    else:
        accelerator, devices = "auto", None

    min_epochs = cfg_trainer.get('min_epochs', 0)
    max_epochs = cfg_trainer.get('max_epochs', -1)
    if cfg_trainer.get('enable_checkpointing', True):
        default_root_dir = config.save_dir
    else:
        default_root_dir = None

    callbacks = []
    if 'callbacks' in cfg_trainer:
        for cb, values in cfg_trainer['callbacks'].items():
            if isinstance(values, list):
                for args in values:
                    callback = getattr(module_callback, cb)(**args)
                    callbacks.append(callback)
            else:
                callback = getattr(module_callback, cb)(**values)
                callbacks.append(callback)
    return lit.Trainer(min_epochs=min_epochs, max_epochs=max_epochs, callbacks=callbacks, accelerator="auto",
                       devices=devices
                       , default_root_dir=default_root_dir)


class CompressionTrainer(lit.Trainer):
    def __init__(self, config, **kwargs):
        self.config = config
        super().__init__(**kwargs)

    # def compute_pruning_amount(self, epoch):
    #     if epoch in config['trainer']
