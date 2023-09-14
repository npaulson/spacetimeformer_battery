import logging
import time
import numpy as np
import pytorch_lightning as pl
import torch.distributions as pyd

import torch
import wandb

class TestOutputCallback(pl.Callback):

    def __init__(self, dataloader,
                 inv_scaler=None, channel_labels=None, dset=None,
                 set_name=None, skip_context=1, skip_target=1):

        loggercore = logging.getLogger('pytorch_lightning.core')
        self.loggercore = loggercore
        self.dataloader = dataloader
        self.channel_labels = channel_labels
        self.dset = dset
        self.set_name = set_name
        self.inv_scaler = inv_scaler
        self.skip_context = skip_context
        self.skip_target = skip_target

    def on_validation_end(self, trainer, model):

        tbl = wandb.Table(columns=['x_c', 'y_c', 'x_t', 'y_t', 'pred'], allow_mixed_types=True)

        for batch in iter(self.dataloader):

            x_c, y_c, x_t, y_t = [i.detach().to(model.device) for i in batch]

            self.loggercore.error('x_c.shape, y_c.shape, x_t.shape, y_t.shape: %s, %s, %s, %s',
                                  x_c.shape, y_c.shape, x_t.shape, y_t.shape)

            if torch.any(torch.isnan(x_c)):
                self.loggercore.error('nan in x_c')
            if torch.any(torch.isnan(y_c)):
                self.loggercore.error('nan in y_c')
            if torch.any(torch.isnan(x_t)):
                self.loggercore.error('nan in x_t')
            if torch.any(torch.isnan(y_t)):
                self.loggercore.error('nan in y_t')

            with torch.no_grad():
                preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)
                if isinstance(preds, pyd.Normal):
                    preds_std = preds.scale.squeeze(-1).cpu().numpy()
                    preds = preds.mean
                else:
                    preds_std = [None for _ in range(preds.shape[0])]
            
            for i in range(preds.shape[0]):

                x_c_ = x_c[i].cpu().numpy()
                y_c_ = y_c[i].cpu().numpy()
                x_t_ = x_t[i].cpu().numpy()
                y_t_ = y_t[i].cpu().numpy()
                preds_ = preds[i].cpu().numpy()

                if 'battery' in self.dset:
                    nonzero = y_c_[..., 0] != 0
                    y_c_ = y_c_[nonzero, :]

                    nonzero = y_t_[..., 0] != 0
                    x_t_ = x_t_[nonzero, :]
                    y_t_ = y_t_[nonzero, :]
                    preds_ = preds_[nonzero, :]

                if self.inv_scaler is not None:
                    y_c_ = self.inv_scaler(y_c_)
                    y_t_ = self.inv_scaler(y_t_)
                    preds_ = self.inv_scaler(preds_)

                tbl.add_data(x_c_, y_c_, x_t_, y_t_, preds_)

        mins = np.round(time.time()/60)

        wandb.log({f'results_{mins}mins_{self.set_name}': tbl})


class TeacherForcingAnnealCallback(pl.Callback):
    def __init__(self, start, end, epochs):
        assert start >= end
        self.start = start
        self.end = end
        self.epochs = epochs
        self.slope = float((start - end)) / epochs

    def on_validation_epoch_end(self, trainer, model):
        current = model.teacher_forcing_prob
        new_teacher_forcing_prob = max(self.end, current - self.slope)
        model.teacher_forcing_prob = new_teacher_forcing_prob

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--teacher_forcing_start", type=float, default=0.8)
        parser.add_argument("--teacher_forcing_end", type=float, default=0.0)
        parser.add_argument("--teacher_forcing_anneal_epochs", type=int, default=8)


class TimeMaskedLossCallback(pl.Callback):
    def __init__(self, start, end, steps):
        assert start <= end
        self.start = start
        self.end = end
        self.steps = steps
        self.slope = float((end - start)) / steps
        self._time_mask = self.start

    @property
    def time_mask(self):
        return round(self._time_mask)

    def on_train_start(self, trainer, model):
        if model.time_masked_idx is None:
            model.time_masked_idx = self.time_mask

    def on_train_batch_end(self, trainer, model, *args):
        self._time_mask = min(self.end, self._time_mask + self.slope)
        model.time_masked_idx = self.time_mask
        model.log("time_masked_idx", self.time_mask)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--time_mask_start", type=int, default=1)
        parser.add_argument("--time_mask_end", type=int, default=12)
        parser.add_argument("--time_mask_anneal_steps", type=int, default=1000)
        parser.add_argument("--time_mask_loss", action="store_true")
