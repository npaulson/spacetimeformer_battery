import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import numpy as np
import core_compute as cc
import core_plot as cp
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def likelihood(params, D):

    if len(params) == 2:
        res = params[0]
    elif len(params) == 3:
        res = params[0] + params[1]*D['x_data']
    else:
        res = None

    prob = ss.norm.logpdf(
        D['y_data']-res,
        loc=0, scale=params[-1]).sum()

    if np.isnan(prob):
        prob = -np.inf

    return prob


def lin_model(trace):

    x_plt = np.linspace(-10, 74, 100)
    pred = x_plt[:, None]*trace[None, :, 1] + trace[:, 0]

    low, med, high = np.percentile(pred, [2.5, 50, 97.5], axis=1)

    return x_plt, low, med, high


if __name__ == '__main__':

    df = pd.read_csv('context_vs_err.csv')
    print(df)
    detail = 'allpts'

    D = {}
    D['outname'] = f'MSE_const_{detail}'
    print('\n\n\n\n', D['outname'])
    D['npar_model'] = 1
    D['distV'] = ['triang', 'triang']
    D['locV'] = [0, 0]
    D['scaleV'] = [.5, .1]
    D['cV'] = [.5, 0]
    D['pname'] = ['b', 'Sigma']
    D['pname_plt'] = D['pname'].copy()
    D['nparam'] = len(D['pname'])
    D['dim'] = len(D['pname'])
    D['likelihood'] = likelihood
    D['x_data'] = df['Context points'].values
    D['y_data'] = df['Loss'].values
    D = cc.sampler_multinest(D)
    print("sampling time: " + str(D['sampling_time']) + " seconds")
    cp.plot_cov(D['rawtrace'], D['pname_plt'], pltname=D['outname'],
                tight_layout=False)

    D['outname'] = f'MAE_const_{detail}'
    print('\n\n\n\n', D['outname'])
    D['y_data'] = df['MAE'].values
    D = cc.sampler_multinest(D)
    print("sampling time: " + str(D['sampling_time']) + " seconds")
    cp.plot_cov(D['rawtrace'], D['pname_plt'], pltname=D['outname'],
                tight_layout=False)

    D['outname'] = f'Loss_lin_{detail}'
    print('\n\n\n\n', D['outname'])
    D['npar_model'] = 2
    D['distV'] = ['triang', 'triang', 'triang']
    D['locV'] = [0, -.1, 0]
    D['scaleV'] = [.5, .2, .1]
    D['cV'] = [.5, .5, 0]
    D['pname'] = ['b', 'm', 'Sigma']
    D['pname_plt'] = D['pname'].copy()
    D['nparam'] = len(D['pname'])
    D['dim'] = len(D['pname'])
    D['y_data'] = df['Loss'].values
    D = cc.sampler_multinest(D)
    trace_mse = D['rawtrace']
    print("sampling time: " + str(D['sampling_time']) + " seconds")
    cp.plot_cov(D['rawtrace'], D['pname_plt'], pltname=D['outname'],
                tight_layout=False)

    D['outname'] = f'MAE_lin_{detail}'
    print('\n\n\n\n', D['outname'])
    D['y_data'] = df['MAE'].values
    D = cc.sampler_multinest(D)
    trace_mae = D['rawtrace']
    print("sampling time: " + str(D['sampling_time']) + " seconds")
    cp.plot_cov(D['rawtrace'], D['pname_plt'], pltname=D['outname'],
                tight_layout=False)

    fig, ax1 = plt.subplots(dpi=200)

    fs = 12

    x_plt, low, med, high = lin_model(trace_mse)
    ax1.fill_between(x_plt, low, high, color='r', alpha=0.1)
    ax1.plot(x_plt, low, 'r:', alpha=0.7)
    ax1.plot(x_plt, med, 'r-', alpha=0.7)
    ax1.plot(x_plt, high, 'r:', alpha=0.7)
    ax1.plot(df['Context points'], df['Loss'], marker='s', ls='', c='r')

    ax1.set_xlim(-4, 69)
    ax1.set_xlabel('number of context points', fontsize=fs)
    ax1.set_ylabel('MSE Loss', color='r', fontsize=fs)

    ax1.tick_params(axis='y', labelcolor='r', labelsize=fs)
    ax1.tick_params(axis='x', labelsize=fs)    

    ax2 = ax1.twinx()

    x_plt, low, med, high = lin_model(trace_mae)
    ax2.fill_between(x_plt, low, high, color='b', alpha=0.1)
    ax2.plot(x_plt, low, 'b:', alpha=0.7)
    ax2.plot(x_plt, med, 'b-', alpha=0.7)
    ax2.plot(x_plt, high, 'b:', alpha=0.7)
    ax2.plot(df['Context points'], df['MAE'], marker='o', ls='', c='b')

    ax2.set_ylabel('MAE Loss', color='b', fontsize=fs)
    ax2.tick_params(axis='y', labelcolor='b', labelsize=fs)
    
    fig.tight_layout()
    fig.savefig(f'context_vs_err_{detail}.png')
