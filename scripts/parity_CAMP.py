import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


if __name__ == '__main__':

    # label for plots
    cset = sys.argv[1]

    # percent fraction of initial capacity for EOL definition
    pct_lvl = np.float64(sys.argv[2])

    # whether to plot outputs
    out = False

    cathodes = ['5VSpinel',  'HE5050', 'NMC111', 'NMC532', 'NMC622']
    cs = sns.color_palette('hls', 8)[:-3]
    ms = ['+', 'x', '<', '^', '>']

    plt.figure(num=f'parity_{pct_lvl}_cap', figsize=(5, 5), dpi=200)

    x_data_all = np.array([])
    y_data_all = np.array([])

    for kk, (cathode, color, marker) in enumerate(zip(cathodes, cs, ms)):
        df = pd.read_csv(f'exp_pred_{pct_lvl}_test_{cathode}.csv')
        x_data = df['exp'].values
        y_data = df['pred'].values
        x_data_all = np.concatenate([x_data_all, x_data])
        y_data_all = np.concatenate([y_data_all, y_data])

    xmin = np.nanmin([np.nanmin(x_data_all), np.nanmin(y_data_all)])
    xmax = np.nanmax([np.nanmax(x_data_all), np.nanmax(y_data_all)])
    print(xmin, xmax)

    xrng = xmax - xmin
    spc = 0.1

    plt.plot([xmin - spc*xrng, xmax + spc*xrng], [xmin - spc*xrng, xmax + spc*xrng], 'k-')

    eol_metrics = {'MAE':[], 'RMSE':[]}

    for kk, (cathode, color, marker) in enumerate(zip(cathodes, cs, ms)):

        df = pd.read_csv(f'exp_pred_{pct_lvl}_test_{cathode}.csv')
        x_data = df['exp'].values
        y_data = df['pred'].values

        mae = np.round(np.nanmean(np.abs(x_data - y_data))).astype('int32')
        label = f'{cathode}'

        plt.plot(x_data, y_data, color=color, mfc='none',
            marker=marker, ls='', ms=5, alpha=0.99, label=label)

        eol_metrics['MAE'].append(np.nanmean(np.abs(x_data-y_data)))
        eol_metrics['RMSE'].append(np.sqrt(np.nanmean((x_data-y_data)**2)))

    plt.xlim(xmin - spc*xrng, xmax + spc*xrng)
    plt.ylim(xmin - spc*xrng, xmax + spc*xrng)

    mae = np.round(np.nanmean(np.abs(x_data_all - y_data_all)), 1)
    rmse = np.round(np.sqrt(np.nanmean((x_data_all - y_data_all)**2)), 1)
    mape = np.round(100*np.nanmean(np.abs(x_data_all - y_data_all)/np.abs(x_data_all)), 1)

    if len(x_data) <= 1:
        r2val = np.nan
    else:
        r2val = 1 - (np.nansum((x_data_all - y_data_all)**2) /
                     np.nansum((x_data_all - np.nanmean(x_data_all))**2))
        if r2val < 0:
            r2val = 0.
        elif r2val > 1:
            r2val = 1.

        r2 = np.round(r2val, 3)

    eol_metrics['MAE'].append(mae)
    eol_metrics['RMSE'].append(rmse)

    index = np.append(cathodes, 'All')
    pd.DataFrame(eol_metrics, index=index).to_csv(f'eol_{cset}_{pct_lvl}.csv')

    plt.xlim(xmin - spc*xrng, xmax + spc*xrng)
    plt.ylim(xmin - spc*xrng, xmax + spc*xrng)

    plt.title(f'MAE: {mae} cycles, RMSE: {rmse} cycles,\nMAPE: {mape}, R2: {r2}')

    plt.xlabel('experiment (cycles)')
    plt.ylabel('prediction (cycles)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{cset}_{pct_lvl}_cap.png')
