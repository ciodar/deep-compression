import operator

import torch
from torch.nn.utils.prune import is_pruned

from trainer.trainer import Trainer

import pruning as module_prune
import quantization as module_quantize
from numpy import inf

class CompressionTrainer(Trainer):
    def __init__(self, model, criterion, metrics, optimizer, config, device, data_loader, **kwargs):
        super().__init__(model, criterion, metrics, optimizer, config, device, data_loader, **kwargs)
        self.pruners = self.config['pruners']
        self.quantizer = self.config['quantizer']

    def prune(self):
        if self.config.resume is None:
            self.train()
        self.logger.info('Accuracy before compression: {:.4f}'.format(self.mnt_best))
        tot_params = 0
        tot_retained = 0
        for pruner in self.pruners:
            self.mnt_best = -inf
            prune_fn = getattr(module_prune, pruner['type'])
            # >1 for iterative pruning
            iterations = 1 if not 'iterations' in pruner else pruner['iterations']
            for it in range(1, iterations + 1):
                p, r = self.prune_model(prune_fn, pruner['levels'])
                tot_params += p
                tot_retained += r
                # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
                trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
                self.optimizer = self.config.init_obj('optimizer', torch.optim, trainable_params)
                self.lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)

                if pruner['finetune_epochs'] > 0:
                    self.start_epoch = 1
                    self.epochs = pruner['finetune_epochs']
                    self.train()
                _, acc1, acc5 = self._valid_epoch(self.start_epoch + 1).values()

                self.logger.info(
                    "Pruning iteration {:d} | acc@1:{:.4f} | acc@5:{:.4f}".format(it, acc1, acc5))

        self.logger.info('Total pruned: {:d} | Retained: {:d} ({:.0%}) | Compression factor: {:.0f}X '.format(
            tot_params - tot_retained, tot_retained, tot_retained / tot_params, tot_params / tot_retained
        ))

    def quantize(self):
        quantize_fn = getattr(module_quantize, self.quantizer['type'])
        self.quantize_model(quantize_fn, self.quantizer['levels'])

        if self.quantizer['finetune_epochs'] > 0:
            self.start_epoch = 1
            self.epochs = self.quantizer['finetune_epochs']
            self.train()

        _, acc1, acc5 = self._valid_epoch(self.start_epoch + 1).values()

        self.logger.info(
            "Tested model after quantization - acc@1:{:.4f} | acc@5:{:.4f}".format(acc1, acc5))



    def prune_model(self, prune_fn, levels):
        for p in self.model.parameters():
            p.requires_grad = False
        tot_params = 0
        tot_retained = 0
        for param, amount in levels.items():
            self.logger.debug('   Pruning {} with amount {:.2f} ...'.format(param, amount))
            # get param name separated from module
            m, param = param.split('.')[0:-1], param.split('.')[-1]
            module = operator.attrgetter('.'.join(m))(self.model)
            prune_fn(module, param, amount)
            # calculate compression stats
            param_vector = torch.nn.utils.parameters_to_vector(module._buffers[param + '_mask'])
            tot_params += param_vector.numel()
            tot_retained += param_vector.nonzero().size(0)
            self.logger.info('   Pruned {} weights - {:d} ({:.2%}) retained)'
                              .format(sum(param_vector == 0).item(),
                                      param_vector.nonzero().size(0),
                                      param_vector.nonzero().size(0) / param_vector.numel()))
            # Always retrain all parameters (eg. bias) even if not pruned
            for p in module.parameters():
                p.requires_grad = True
        return tot_params, tot_retained

    def quantize_model(self, quantize_fn, levels):
        for p in self.model.parameters():
            p.requires_grad = False
        for param, bits in levels.items():
            self.logger.debug('   Quantizing {} in {:d} bits ...'.format(param, bits))
            # get param name separated from module
            m, param = param.split('.')[0:-1], param.split('.')[-1]
            module = operator.attrgetter('.'.join(m))(self.model)
            quantize_fn(module, param, bits)
            # Retrain all parameters of quantized model (also bias) even if not pruned
            for p in module.parameters():
                p.requires_grad = True
