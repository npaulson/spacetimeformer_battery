import sys
import json
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


if __name__ == '__main__':

    # fname: file in form of results_*mins_test_*table.json
    # on wandb, 'runs/<project>/root/media/table'
    fname = sys.argv[1]

    # fname_inp_lab: file in form of inputs_labels_battery_*.table.json
    # on wandb, 'runs/<project>/root/media/table'
    fname_inp_lab = sys.argv[2]

    # whether to plot extrapolations
    out = False

    # plotting interval
    pltint = 100

    with open(fname) as json_file:
        data = json.load(json_file)
    with open(fname_inp_lab) as json_file:
        inp_lab = json.load(json_file)

    print('columns:', data['columns'])

    n_tst = len(data['data'])

    print("n_tst:", n_tst)

    names = [
        'discharge capacity', 'discharge energy',
        'discharge avg. V', 'Q_eff', 'E_eff',
        'R_10', 'OCV_10', 'R_20', 'OCV_20',
        'R_30', 'OCV_30', 'R_40', 'OCV_40',
        'R_50', 'OCV_50', 'R_60', 'OCV_60',
        'R_70', 'OCV_70', 'R_80', 'OCV_80',
        'R_90', 'OCV_90']

    for ii in range(0, n_tst, pltint):

        label_info = inp_lab['data'][ii]
        label_info = [str(lab) for lab in label_info]
        label_info = '_'.join(label_info)

        item = data['data'][ii]
        x_c = np.array(item[0])
        y_c = np.array(item[1])
        x_t = np.array(item[2])
        y_t = np.array(item[3])
        pred = np.array(item[4])

        print('x_c.shape:', x_c.shape,
              'y_c.shape:', y_c.shape,
              'x_t.shape:', x_t.shape,
              'y_t.shape:', y_t.shape,
              'pred.shape:', pred.shape)

        cycles_c = x_c[:, 0]
        cycles_t = x_t[:, 0]

        sel = cycles_t != 0
        print(sel.shape)
        cycles_t = cycles_t[sel]
        y_t = y_t[sel, :]
        pred = pred[sel, :]

        nout = pred.shape[1]

        ncol = np.ceil(np.sqrt(nout)).astype('int8')
        nrow = np.ceil(nout/ncol).astype('int8')

        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*3), 
                                squeeze=False, dpi=200)
        fs = 11
        lw = 1
        pad = 1
        tl = 1

        for jj in range(nout):

            aa, bb = np.unravel_index(jj, (nrow, ncol))

            axs[aa, bb].plot(
                cycles_c, y_c[:, jj], 'bo-', ms=3, lw=lw, label='context')
            axs[aa, bb].plot(
                cycles_t, y_t[:, jj], 'ks-', ms=3, lw=lw, label='experiment')
            axs[aa, bb].plot(
                cycles_t, pred[:, jj], 'r.-', ms=3, lw=lw, alpha=0.5, label='prediction')
            axs[aa, bb].tick_params(axis='both', which='both', labelsize=fs, length=1)
            axs[aa, bb].set_xlabel('cycle number', fontsize=fs+3)
            axs[aa, bb].set_ylabel(names[jj], fontsize=fs+3)
            if jj == 0:            
                axs[aa, bb].legend(fontsize=fs)

        for jj in range(nout, nrow*ncol):
            plt.subplot(nrow, ncol, jj+1).axis('off') 

        fig.tight_layout(h_pad=pad, w_pad=pad)
        fig.savefig(f'{label_info}_prognosis.png')
        plt.close()
