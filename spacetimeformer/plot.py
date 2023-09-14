import io
import os

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.distributions as pyd
import pandas as pd
import cv2
import random
import torch
import wandb

from spacetimeformer.eval_stats import mape


def _assert_squeeze(x):
    assert len(x.shape) == 2
    return x.squeeze(-1)


def plot(x_c, y_c, x_t, y_t, preds,
         conf=None, inv_scaler=None,
         channel_labels=None, dset=None,
         skip_context=1, skip_target=1):

    if 'battery' in dset:
        nonzero = y_c[..., 0] != 0
        y_c = y_c[nonzero, :]

        nonzero = y_t[..., 0] != 0
        y_t = y_t[nonzero, :]
        preds = preds[nonzero, :]

    if inv_scaler is not None:
        y_c = inv_scaler(y_c)
        y_t = inv_scaler(y_t)
        preds = inv_scaler(preds)

    imgs = []

    for idx in range(y_c.shape[-1]):

        y_c_i = y_c[..., idx]
        y_t_i = y_t[..., idx]
        preds_i = preds[..., idx]

        fig, ax = plt.subplots(figsize=(7, 4))
        xaxis_c = np.arange(0, skip_context*len(y_c_i), skip_context)
        x_t_st = xaxis_c[-1] + skip_target
        xaxis_t = np.arange(x_t_st, x_t_st + skip_target*len(y_t_i), skip_target)
        context = pd.DataFrame({"xaxis_c": xaxis_c, "y_c": y_c_i})
        target = pd.DataFrame({"xaxis_t": xaxis_t, "y_t": y_t_i, "pred": preds_i})
        ax.scatter(
            x=target["xaxis_t"], y=target["y_t"], c="grey", label="True",
            marker='o', linewidth=1.0
        )
        ax.scatter(
            x=target['xaxis_t'], y=target['pred'], c='red', label='Forecast',
            marker='.', linewidth=1)

        ax.scatter(
            x=context["xaxis_c"], y=context["y_c"], c="blue", label="Context",
            marker='s')

        ax.legend(prop={"size": 12})
        ax.set_facecolor("#f0f0f0")

        if dset == 'battery':
            ax.set_xlabel("cycles")
        else:
            ax.set_xlabel("")

        if channel_labels is not None:
            ax.set_ylabel(channel_labels[idx])
        else:
            ax.set_ylabel("")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=128)

        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgs.append(img)

    return imgs


class PredictionPlotterCallback(pl.Callback):

    def __init__(self, test_batches, total_samples=4,
                 inv_scaler=None, channel_labels=None, dset=None,
                 set_name=None, skip_context=1, skip_target=1):
        self.test_data = test_batches
        self.total_samples = total_samples
        self.channel_labels = channel_labels
        self.dset = dset
        self.set_name = set_name
        self.inv_scaler = inv_scaler
        self.skip_context = skip_context
        self.skip_target = skip_target

    def on_validation_end(self, trainer, model):

        idxs = [random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)]

        for ii in range(self.test_data[3].shape[0]):
            tst = self.test_data[3][ii, ..., 0]

        x_c, y_c, x_t, y_t = [i[idxs].detach().to(model.device) for i in self.test_data]
        with torch.no_grad():
            preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)
            if isinstance(preds, pyd.Normal):
                preds_std = preds.scale.squeeze(-1).cpu().numpy()
                preds = preds.mean
            else:
                preds_std = [None for _ in range(preds.shape[0])]
        
        i = random.randrange(0, preds.shape[0])
        imgs_ = plot(
            x_c[i].cpu().numpy(),
            y_c[i].cpu().numpy(),
            x_t[i].cpu().numpy(),
            y_t[i].cpu().numpy(),
            preds[i].cpu().numpy(),
            conf=preds_std[i],
            inv_scaler = self.inv_scaler,
            channel_labels=self.channel_labels,
            dset=self.dset,
            skip_context=self.skip_context,
            skip_target=self.skip_target
        )

        imgs = []
        for img in imgs_:
            if img is not None:
                imgs.append(wandb.Image(img, caption=''))

        trainer.logger.experiment.log(
            {
                f"{self.set_name}/prediction_plots": imgs,
                "global_step": trainer.global_step,
            }
        )


def attn_plot(attn, title, tick_spacing=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.imshow(attn.cpu().numpy(), cmap="Blues")
    if tick_spacing:
        plt.xticks(np.arange(0, attn.shape[0] + 1, tick_spacing))
        plt.yticks(np.arange(0, attn.shape[0] + 1, tick_spacing))

    plt.title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=128)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AttentionMatrixCallback(pl.Callback):
    def __init__(self, test_batches, layer=0, total_samples=32, raw_data_dir=None):
        self.test_data = test_batches
        self.total_samples = total_samples
        self.layer = layer
        self.raw_data_dir = raw_data_dir

    def on_validation_end(self, trainer, model):
        with torch.no_grad():
            idxs = [
                random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)
            ]
            x_c, y_c, x_t, y_t = [
                i[idxs].detach().to(model.device) for i in self.test_data
            ]
            attns = None
            # save memory by doing inference 1 example at a time
            for i in range(self.total_samples):
                x_ci = x_c[i].unsqueeze(0)
                y_ci = y_c[i].unsqueeze(0)
                x_ti = x_t[i].unsqueeze(0)
                y_ti = y_t[i].unsqueeze(0)
                *_, attn = model(x_ci, y_ci, x_ti, y_ti, output_attn=True)
                if attns is None:
                    attns = [[a] for a in attn]
                else:
                    for cum_attn, attn in zip(attns, attn):
                        cum_attn.append(attn)
            # re-concat over batch dim
            attns = [torch.cat(a, dim=0) for a in attns]
            # average over batch dim
            attn = attns[self.layer].mean(0)

        heads = [i for i in range(attn.shape[0])] + ["avg", "sum"]

        imgs = []
        for head in heads:
            if head == "avg":
                a_head = attn.mean(0)
            elif head == "sum":
                a_head = attn.sum(0)
            else:
                a_head = attn[head]

            a_head = (a_head - a_head.mean()) / (a_head.std() + 1e-5)
            img = wandb.Image(attn_plot(a_head, str(head), tick_spacing=y_c.shape[-2]))
            imgs.append(img)

        trainer.logger.experiment.log(
            {
                "test/attn": imgs,
                "global_step": trainer.global_step,
            }
        )

        if self.raw_data_dir is not None:
            np.savez(
                os.path.join(self.raw_data_dir, "attn_matrix.npz"),
                attn=attn.cpu().numpy(),
            )
