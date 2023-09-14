# Prognosis of Multivariate Battery State of Performance and Health via Transformers - Repository

This repository contains all codes required to reproduce the results in the manuscript (currently under submission - a link will be provided when available). To run the spacetimeformer model, first install the package as described in the installation instructions below. Then download both directories from the ([materials data facility entry](https://doi.org/10.18126/ckt2-g8j2)) and put them in the ./spacetimeformer_battery/spacetimeformer/data/ directory. Below is an example of how to run the model for the CAMP dataset:

```
>>> python train.py spacetimeformer battery_CAMP --run_name spatiotemporal_battery_CAMP --d_model 100 --d_ff 200 --enc_layers 4 --dec_layers 4 --gpus 0 --batch_size 4 --start_token_len 4 --n_heads 4 --context_points 1 --skip_context 1 --skip_target 10 --early_stopping --wandb --plot
```

In order, use python to call 'train.py' with the spacetimeformer model, battery_CAMP dataset, and the specified hyperparameters. Add or remove the last three entries to control other elements of the code behavior.

Once a run has been performed and the results automatically uploaded to ([Weights and Biases](https://wandb.ai/site)), relevant data files can be downloaded to generate the figures used in the manuscript with the python scripts in the ./spacetimeformer_battery/scripts/ directory.

Below is the original readme from late January 2022. Many thanks to the original spacetimeformer team for making their code available and easy to use.

# Spacetimeformer Multivariate Forecasting

This repository contains the code for the paper, "**Long-Range Transformers for Dynamic Spatiotemporal Forecasting**", Grigsby, Wang and Qi, 2021. ([arXiv](https://arxiv.org/abs/2109.12218))

![spatiotemporal_embedding](readme_media/st-graph.png)

Transformers are a high-performance approach to sequence-to-sequence timeseries forecasting. However, stacking multiple sequences into each token only allows the model to learn *temporal* relationships across time. This can ignore important *spatial* relationships between variables. Our model (nickamed "Spacetimeformer") flattens multivariate timeseries into extended sequences where each token represents the value of one variable at a given timestep. Long-Range Transformers can then learn relationships over both time and space. For much more information, please refer to our paper.

## Installation 
This repository was written and tested for **python 3.8** and **pytorch 1.9.0**.

```bash
git clone https://github.com/QData/spacetimeformer.git
cd spacetimeformer
conda create -n spacetimeformer python==3.8
source activate spacetimeformer
pip install -r requirements.txt
pip install -e .
```
This installs a python package called ``spacetimeformer``.

## Dataset Setup
CSV datsets like AL Solar, NY-TX Weather, Exchange Rates, and the Toy example are included with the source code of this repo. 

Larger datasets should be downloaded and their folders placed in the `data/` directory. You can find them with this [google drive link](https://drive.google.com/drive/folders/1NcCIjuWbkvAi1MZUpYBIr7eYhaowvU7B?usp=sharing). Note that the `metr-la` and `pems-bay` data is directly from [this repo](https://github.com/liyaguang/DCRNN) - all we've done is skip a step for you and provide the raw train, val, test, `*.npz` files our dataset code expects.


## Recreating Experiments with Our Training Script
The main training functionality for `spacetimeformer` and most baselines considered in the paper can be found in the `train.py` script. The training loop is based on the [`pytorch_lightning`](https://pytorch-lightning.rtfd.io/en/latest/) framework.

Commandline instructions for each experiment can be found using the format: ```python train.py *model* *dataset* -h```. 

Model Names:
- `linear`: a basic autoregressive linear model.
- `lstnet`: a more typical RNN/Conv1D model for multivariate forecasting. Based on the attention-free implementation of [LSTNet](https://github.com/laiguokun/LSTNet).
- `lstm`: a typical encoder-decoder LSTM without attention. We use scheduled sampling to anneal teacher forcing throughout training.
- `mtgnn`: a hybrid GNN that learns its graph structure from data. For more information refer to the [paper](https://arxiv.org/abs/2005.11650). We use the implementation from [`pytorch_geometric_temporal`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
- `spacetimeformer`: the multivariate long-range transformer architecture discussed in our paper.
    - note that the "Temporal" ablation discussed in the paper is a special case of the `spacetimeformer` model. Set the `embed_method = temporal`. Spacetimeformer has many configurable options and we try to provide a thorough explanation with the commandline `-h` instructions.


Dataset Names:
- `metr-la` and `pems-bay`: traffic forecasting datasets. We use a very similar setup to [DCRNN](https://github.com/liyaguang/DCRNN).
- `toy2`: is the toy dataset mentioned at the beginning of our experiments section. It is heavily based on the toy dataset in [TPA-LSTM](https://arxiv.org/abs/1809.04206.).
- `asos`: Is the codebase's name for what the paper calls "NY-TX Weather."
- `solar_energy`: Is the codebase's name for what is more commonly called "AL Solar."
- `exchange`: A dataset of exchange rates. Spacetimeformer performs relatively well but this is tiny dataset of highly non-stationary data where `linear` is already a SOTA model.
- `precip`: A challenging spatial message-passing task that we have not yet been able to solve. We collected daily precipitation data from a latitude-longitude grid over the Continental United States. The multivariate sequences are sampled from a ringed "radar" configuration as shown below in green. We expand the size of the dataset by randomly moving this radar around the country.

<p align="center">
<img src="readme_media/radar_edit.png" width="220">
</p>

### Logging with Weights and Biases
We used [wandb](https://wandb.ai/home) to track all of results during development, and you can do the same by providing your username and project as environment variables:
```bash
export STF_WANDB_ACCT="your_username"
export STF_WANDB_PROJ="your_project_title"
# optionally: change wandb logging directory (defaults to ./data/STF_LOG_DIR)
export STF_LOG_DIR="/somewhere/with/more/disk/space"
```
wandb logging can then be enabled with the `--wandb` flag.

There are two automated figures that can be saved to wandb between epochs. These include the attention diagrams (e.g., Figure 4 of our paper) and prediction plots (e.g., Figure 6 of our paper). Enable attention diagrams with `--attn_plot` and prediction curves with `--plot`.

### Example Spacetimeformer Training Commands
Toy Dataset
```bash
python train.py spacetimeformer toy2 --run_name spatiotemporal_toy2 \
--d_model 100 --d_ff 400 --enc_layers 4 --dec_layers 4 \
--gpus 0 1 2 3 --batch_size 32 --start_token_len 4 --n_heads 4 \
--grad_clip_norm 1 --early_stopping --trials 1
```

Metr-LA
```bash
python train.py spacetimeformer metr-la --start_token_len 3 --batch_size 32 \
--gpus 0 1 2 3 --grad_clip_norm 1 --d_model 128 --d_ff 512 --enc_layers 5 \
--dec_layers 4 --dropout_emb .3 --dropout_ff .3 --dropout_qkv 0 \ 
--run_name spatiotemporal_metr-la --base_lr 1e-3 --l2_coeff 1e-2 \
```

Temporal Attention Ablation with Negative Log Likelihood Loss on NY-TX Weather ("asos") with WandB Logging and Figures
```bash
python train.py spacetimeformer asos --context_points 160 --target_points 40 \ 
--start_token_len 8 --grad_clip_norm 1 --gpus 0 --batch_size 128 \ 
--d_model 200 --d_ff 800 --enc_layers 3 --dec_layers 3 \
--local_self_attn none --local_cross_attn none --l2_coeff .01 \
--dropout_emb .1 --run_name temporal_asos_160-40-nll --loss nll \
--time_resolution 1 --dropout_ff .2 --n_heads 8 --trials 3 \ 
--embed_method temporal --early_stopping --wandb --attn_plot --plot
```




## Using Spacetimeformer in Other Applications
If you want to use our model in the context of other datasets or training loops, you will probably want to go a step lower than the `spacetimeformer_model.Spacetimeformer_Forecaster` pytorch-lightning wrapper. Please see `spacetimeformer_model.nn.Spacetimeformer`.
![arch-fig](readme_media/arch.png)

## Citation
If you use this model in academic work please feel free to cite our paper

```
@misc{grigsby2021longrange,
      title={Long-Range Transformers for Dynamic Spatiotemporal Forecasting}, 
      author={Jake Grigsby and Zhe Wang and Yanjun Qi},
      year={2021},
      eprint={2109.12218},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

![st-embed-fig](readme_media/embed.png)

## Roadmap, V2 Plans

We are working on a second version of the paper, where we plan to focus on adjustments that make it easier to work with real-world datasets:
- [ ] Missing data in the encoder sequence (instead of only ignoring the loss values in the decoder)
- [ ] Multivariate datasets with variables sampled at different time intervals
- [ ] Additional encoder sequence features beyond the target variables

If you have other suggestions, please feel free to file an issue or email the authors!











