import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


if __name__ == '__main__':

    # fname: file in form of results_*mins_test_*table.json
    # on wandb, 'runs/<project>/root/media/table'
    fname = sys.argv[1]

    # fname_inp_lab: file in form of inputs_labels_battery_*.table.json
    # on wandb, 'runs/<project>/root/media/table'
    fname_inp_lab = sys.argv[2]

    # label for the plots
    cset = sys.argv[3]

    # capacity cutoff threshold for EOL
    pct_lvl = np.float32(sys.argv[4])

    # whether to plot extrapolations
    out = False

    with open(fname) as json_file:
        data = json.load(json_file)

    with open(fname_inp_lab) as json_file:
        inp_lab = json.load(json_file)

    inp_lab = pd.DataFrame(inp_lab['data'], columns=inp_lab['columns'])

    cat = []
    for cellname in inp_lab['cellname'].values:
        indx = cellname.find('_')
        cat.append(cellname[:indx])

    inp_lab['category'] = cat

    cs = sns.color_palette('hls', 6)[:-2]
    plt.figure(num = 'histograms')
    categories = np.unique(inp_lab['category'])
    lifes = []
    for category in categories:
        sel = inp_lab['category'] == category
        lifes.append(inp_lab['label_end'].values[sel.values])
    plt.hist(lifes, bins=20, color=cs, alpha=0.8, density=True, label=categories)
    plt.xlabel('end_cycle')
    plt.ylabel('probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cyc_life_hist.png')
    plt.show()

    n_tst = len(data['data'])

    print(f"n_tst: {n_tst}, n_tst inp_lab: {len(inp_lab)}")

    feature_names = [
        'discharge capacity', 'discharge energy',
        'discharge avg. V', 'Q_eff', 'E_eff',
        'R_10', 'OCV_10', 'R_20', 'OCV_20',
        'R_30', 'OCV_30', 'R_40', 'OCV_40',
        'R_50', 'OCV_50', 'R_60', 'OCV_60',
        'R_70', 'OCV_70', 'R_80', 'OCV_80',
        'R_90', 'OCV_90']

    n_feat = len(feature_names)

    n_bin_st = 50
    n_seq = 250
    err = np.nan*np.ones((n_feat, n_tst, n_seq))
    err_st = np.nan*np.ones((n_feat, n_tst, n_bin_st))
    cyc_Xpct_exp = np.zeros((n_tst,))
    cyc_Xpct_pred = np.zeros((n_tst,))

    # create a dictionary for the error metrics
    metric_names = ['MAE', 'RMSE', 'MAPE', 'RMSPE', 'R2']
    metrics = {}
    for metric_name in metric_names:
        metrics[metric_name] = {}
        for feature_name in feature_names:
            metrics[metric_name][feature_name] = []

    r2u = 0

    for ii in range(0, n_tst, 1):

        # these negative values only apply for Severson

        item = data['data'][ii]
        x_c = np.array(item[0])
        y_c = np.array(item[1])
        x_t = np.array(item[2])
        y_t = np.array(item[3])
        pred = np.array(item[4])

        if out:
            print('iter:', ii,
                  'x_c.shape:', x_c.shape,
                  'y_c.shape:', y_c.shape,
                  'x_t.shape:', x_t.shape,
                  'y_t.shape:', y_t.shape,
                  'pred.shape:', pred.shape)

        cond1 = len(x_t.shape) < 2
        cond2 = x_t.shape[0] < 2 or y_t.shape[0] < 2

        if cond1 or cond2:
            for jj, feature_name in enumerate(feature_names):
                metrics['MAE'][feature_name].append(np.nan)
                metrics['RMSE'][feature_name].append(np.nan)
                metrics['MAPE'][feature_name].append(np.nan)
                metrics['RMSPE'][feature_name].append(np.nan)
                metrics['R2'][feature_name].append(np.nan)
                cyc_Xpct_exp[ii] = np.nan
                cyc_Xpct_pred[ii] = np.nan
            continue

        cycles_c = x_c[:, 0]
        cycles_t = x_t[:, 0]
        cycle_st = cycles_t[0] + 10
        cycles = np.round(cycles_t - cycle_st)[:-1]

        if np.abs(10*len(pred) - (inp_lab['label_end'].values[ii]-inp_lab['label_st'].values[ii])) >= 10:
            print(len(pred), inp_lab['label_end'].values[ii]-inp_lab['label_st'].values[ii])

        bin_st = np.int8(np.floor(cycle_st/50))

        # calculate error in predicting cycles to 90% capacity
        init_cap = y_c[0, 0]

        lvl = pct_lvl*init_cap

        cyc_exp_f = interp1d(y_t[:, 0], cycles_t,
                             bounds_error=False, fill_value=np.nan)
        cyc_exp = cyc_exp_f(lvl)

        def life_pred_f(Q_val, Qs, cycs):

            func = interp1d(Qs, cycs,
                            bounds_error=False, fill_value=np.nan)
            cyc_val = func(Q_val)
            pred_type = 'interpolated'
            slope = np.nan

            if np.isnan(cyc_exp):
                return cyc_val, pred_type, slope

            if np.isnan(cyc_val):                

                lr_x = Qs[:, None]
                lr_y = cycs - cycs[-1]
                weights = np.linspace(0.01, 1, len(cycs))

                # regression on last lr_cycs or fewer cycles
                lr_cycs = 10
                if len(cycs) > lr_cycs:
                    lr_x = lr_x[-lr_cycs:, :]
                    lr_y = lr_y[-lr_cycs:]
                    weights = np.linspace(0.01, 1, lr_cycs)

                lr = LinearRegression().fit(
                    lr_x, lr_y, sample_weight=weights)
                slope, intercept = lr.coef_[0], lr.intercept_

                max_slope = -0.1*init_cap*1000.
                min_slope = -10*init_cap*1000.

                if slope > max_slope:
                    slope = min_slope
                elif slope < min_slope:
                    slope = min_slope

                cyc_val = slope*(Q_val - Qs[-1]) + cycs[-1]
                pred_type = 'extrapolated'

            return cyc_val, pred_type, slope

        cyc_model, pred_type_model, slope_model = life_pred_f(lvl, pred[:, 0], cycles_t)

        if cyc_model < 0:
            cyc_model = 0

        # if out and not np.isnan(cyc_exp):
        # if out and pred_type_model == 'extrapolated':
        if out and np.abs(cyc_model-cyc_exp) > 200:

            plt.figure(num='ext_demo')
            plt.plot(y_t[:, 0], cycles_t, 'b.', label='exp.')
            plt.plot(pred[:, 0], cycles_t, 'k.', label='model')

            if not np.isnan(slope_model):
                caps = np.array([init_cap, lvl])
                cycles_m = slope_model*(caps-pred[-1, 0]) + cycles_t[-1]

                ymin = np.min([np.min(cycles_t), np.min(cycles_m)])
                ymax = np.max([np.max(cycles_t), np.max(cycles_m)])
                plt.plot(caps, cycles_m,
                         'k:', label='model extrapolate')
            else:
                ymin = np.min(cycles_t)
                ymax = np.max(cycles_t)

            rng = ymax - ymin
            ymin -= 0.1*rng
            ymax += 0.1*rng

            plt.vlines(lvl, ymin, ymax, colors='r', label='failure capacity (Ah)')
            plt.plot(lvl, cyc_exp, 'bo')
            plt.plot(lvl, cyc_model, 'ks')

            plt.ylim(ymin, ymax)
            plt.xlabel('capacity (Ah)')
            plt.ylabel('cycles')
            plt.legend()
            plt.tight_layout()
            plt.show()

        cyc_Xpct_exp[ii] = cyc_exp
        cyc_Xpct_pred[ii] = cyc_model

        if y_t.shape[0] <= 2:
           r2u += 1

        for jj, feature_name in enumerate(feature_names):
            y = y_t[:-1, jj]
            pred_ = pred[:-1, jj]
            err_ = 100*np.abs(y - pred_)/np.abs(y)
            err[jj, ii, :len(err_)] = err_
            err_st[jj, ii, bin_st] = np.mean(err_)

            metrics['MAE'][feature_name].append(np.nanmean(np.abs(y - pred_)))
            metrics['RMSE'][feature_name].append(np.sqrt(np.nanmean((y - pred_)**2)))
            metrics['MAPE'][feature_name].append(100*np.nanmean(np.abs(y - pred_)/y))
            metrics['RMSPE'][feature_name].append(100*np.sqrt(np.nanmean((y - pred_)/y)**2))

            if len(y) <= 1:
                r2val = np.nan
            else:
                r2val = 1 - (np.sum((y - pred_)**2) /
                             np.sum((y - np.nanmean(y))**2))
                if r2val < 0:
                    r2val = 0
                elif r2val > 1:
                    r2val = 1
            metrics['R2'][feature_name].append(r2val)

    print(f'number of undefined R2 scores: {r2u}')

    plt.figure(num=f'parity_{pct_lvl}_cap', figsize=(5, 5), dpi=200)
    x_data = cyc_Xpct_exp
    y_data = cyc_Xpct_pred

    frac = np.sum(np.invert(np.isnan(x_data)))/n_tst
    print(f'fraction of cells reaching {pct_lvl} of original capacity: {frac}')

    xmin = np.min([np.nanmin(x_data), np.nanmin(y_data)])
    xmax = np.max([np.nanmax(x_data), np.nanmax(y_data)])
    xrng = xmax - xmin
    spc = 0.1

    plt.plot([xmin - spc*xrng, xmax + spc*xrng], [xmin - spc*xrng, xmax + spc*xrng], 'k-')

    cs = sns.color_palette('hls', 8)[:-4]
    ms = ['+', 'x', '<', '^']

    categoriesu = np.unique(inp_lab['category'])
    print(f'categoriesu: {categoriesu}')

    for kk, cat in enumerate(categoriesu):

        sel = (inp_lab['category'].values == cat).astype('bool')

        plt.plot(x_data[sel], y_data[sel], color=cs[kk], mfc='none',
            marker=ms[kk], ls='', ms=5, alpha=0.99, label=cat)

    mae = np.round(np.nanmean(np.abs(x_data - y_data)), 1)
    rmse = np.round(np.sqrt(np.nanmean((x_data - y_data)**2)), 1)
    mape = np.round(100*np.nanmean(np.abs(x_data - y_data)/np.abs(x_data)), 1)

    if len(x_data) <= 1:
        r2val = np.nan
    else:
        r2val = 1 - (np.nansum((x_data - y_data)**2) /
                     np.nansum((x_data - np.nanmean(x_data))**2))
        if r2val < 0:
            r2val = 0.
        elif r2val > 1:
            r2val = 1.
    
        r2 = np.round(r2val, 3)

    plt.xlim(xmin - spc*xrng, xmax + spc*xrng)
    plt.ylim(xmin - spc*xrng, xmax + spc*xrng)

    plt.title(f'MAE: {mae} cycles, RMSE: {rmse} cycles,\nMAPE: {mape}, R2: {r2}')
    plt.xlabel('experiment (cycles)')
    plt.ylabel('prediction (cycles)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{cset}_parity_{pct_lvl}_cap.png')

    """get metric CSV files"""

    metrics_mean = np.zeros((5, n_feat))
    metrics_std = np.zeros((5, n_feat))

    metrics_str = {}
    for ii, metric_name in enumerate(metric_names):
        metrics_str[metric_name] = []
        for jj, feature_name in enumerate(feature_names):

            metrics_mean[ii, jj] = np.nanmean(
                np.array(metrics[metric_name][feature_name]))
            metrics_std[ii, jj] = np.nanstd(
                np.array(metrics[metric_name][feature_name]))

            if metrics_mean[ii, jj] > 100:
                dig = 0
            elif metrics_mean[ii, jj] > 10:
                dig = 1
            elif metrics_mean[ii, jj] > 1:
                dig = 2
            else:
                dig = 3
            mean_rnd = np.round(metrics_mean[ii, jj], dig)
            std_rnd = np.round(metrics_std[ii, jj], dig)
            metstr = f'{mean_rnd} +/- {std_rnd}'
            metrics_str[metric_name].append(metstr)

    df_met = pd.DataFrame(metrics_str, index=feature_names, dtype='string')
    df_met.to_csv(f'{cset}_all_metrics.csv')

    for kk, cat in enumerate(categoriesu):

        sel = (inp_lab['category'].values == cat).astype('bool')

        metrics_mean = np.zeros((5, n_feat))
        metrics_std = np.zeros((5, n_feat))

        metrics_str = {}
        for ii, metric_name in enumerate(metric_names):
            metrics_str[metric_name] = []
            for jj, feature_name in enumerate(feature_names):

                metrics_mean[ii, jj] = np.nanmean(
                    np.array(metrics[metric_name][feature_name])[sel])
                metrics_std[ii, jj] = np.nanstd(
                    np.array(metrics[metric_name][feature_name])[sel])

                if metrics_mean[ii, jj] > 100:
                    dig = 0
                elif metrics_mean[ii, jj] > 10:
                    dig = 1
                elif metrics_mean[ii, jj] > 1:
                    dig = 2
                else:
                    dig = 3
                mean_rnd = np.round(metrics_mean[ii, jj], dig)
                std_rnd = np.round(metrics_std[ii, jj], dig)
                metstr = f'{mean_rnd} +/- {std_rnd}'
                metrics_str[metric_name].append(metstr)

        df_met = pd.DataFrame(metrics_str, index=feature_names, dtype='string')
        df_met.to_csv(f'{cset}_{cat}_metrics.csv')

    """plot error vs cycles"""

    for jj, feature_name in enumerate(feature_names):
        pcntl = np.nanpercentile(
            err[jj, ...], [2.5, 15, 50, 85, 97.5], axis=0).T
        pcntl = pcntl[np.invert(np.isnan(pcntl[:, 0])), :]

        plt.figure(num=f'{feature_name}_err_vs_cyc', figsize=(5, 4), dpi=300)
        cycles = np.arange(10, 10*(pcntl.shape[0]+1), 10)
        plt.semilogy(cycles, pcntl[:, 0], 'k:', label='2.5-97.5th percentiles')
        plt.semilogy(cycles, pcntl[:, 1], 'b--', label='15-85th percentiles')
        plt.semilogy(cycles, pcntl[:, 2], 'r-', label='50th percentile')
        plt.semilogy(cycles, pcntl[:, 3], 'b--')
        plt.semilogy(cycles, pcntl[:, 4], 'k:')
        plt.xlabel('cycles')
        plt.ylabel(f'{feature_name}\nprediction error (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{cset}_{feature_name}_err_vs_cyc.png')
        plt.close()

        plt.figure(num=f'{feature_name}_err_vs_cyc_vs_st', figsize=(5, 4), dpi=300)
        pcntl = np.nanpercentile(
            err_st[jj, ...], [2.5, 15, 50, 85, 97.5], axis=0).T
        sel = np.invert(np.isnan(pcntl[:, 0]))
        pcntl = pcntl[sel, :]
        cycles = np.arange(0, 50*pcntl.shape[0], 50)
        plt.semilogy(cycles, pcntl[:, 0], 'k:', label='2.5-97.5th percentiles')
        plt.semilogy(cycles, pcntl[:, 1], 'b--', label='15-85th percentiles')
        plt.semilogy(cycles, pcntl[:, 2], 'r-', label='50th percentile')
        plt.semilogy(cycles, pcntl[:, 3], 'b--')
        plt.semilogy(cycles, pcntl[:, 4], 'k:')
        plt.xlabel('start cycle')
        plt.ylabel(f'{feature_name}\nprediciton error (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{cset}_{feature_name}_err_vs_cyc_vs_st.png')
        plt.close()
