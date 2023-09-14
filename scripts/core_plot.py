import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import cumtrapz


def plot_chains(rawtrace, flattrace, nlinks, pname, pname_plt, pltname=None):

    nwalkers = rawtrace.shape[0]
    nlinksT = rawtrace.shape[1]
    npar = rawtrace.shape[2]

    ntune = nlinksT-nlinks

    for ii in range(npar):

        plt.figure(figsize=[4, 2.5])

        for jj in range(nwalkers-2):
            plt.plot(range(nlinksT), rawtrace[jj, :, ii],
                     linestyle='-', marker='', color='b',
                     lw=.5, alpha=.15)

        for jj in range(nwalkers-2, nwalkers):
            plt.plot(range(nlinksT), rawtrace[jj, :, ii],
                     linestyle='-', marker='', color='k',
                     lw=.5, alpha=.8)

        plt.axvline(ntune, color="k", linestyle=':')

        plt.xlabel('iteration number')
        plt.ylabel(r'$%s$' % pname_plt[ii], fontsize=15)

        plt.tight_layout()

        if pltname is not None:
            plt.savefig(pltname + "_chain" + pname[ii] + ".png")
            plt.close()


def plot_cov(flattrace, pname_plt, sciform=False, param_true=None,
             bounds=None, pltname=None, tight_layout=True, figsize=[9, 9]):

    plt.figure(figsize=figsize)

    n_par = len(pname_plt)

    rindx = np.arange(flattrace.shape[0])
    rindx = np.random.choice(rindx, 2000)

    for ii in range(n_par):
        ftii = flattrace[:, ii]

        if bounds is not None:
            iist = bounds[ii][0]
            iiend = bounds[ii][1]
        else:
            iimin = ftii.min()
            iimax = ftii.max()
            iirng = iimax-iimin
            iist = iimin-.1*iirng
            iiend = iimax+.1*iirng

        for jj in range(ii+1):
            ftjj = flattrace[:, jj]

            if bounds is not None:
                jjst = bounds[jj][0]
                jjend = bounds[jj][1]
            else:
                jjmin = ftjj.min()
                jjmax = ftjj.max()
                jjrng = jjmax-jjmin
                jjst = jjmin-.1*jjrng
                jjend = jjmax+.1*jjrng

            ax = plt.subplot(n_par, n_par, ii*n_par+jj+1)

            if ii == jj:
                sns.distplot(ftii, bins=15)
                if param_true is not None:
                    ax.axvline(param_true[ii], linestyle=':', color="k")
                plt.xlim(iist, iiend)
            else:
                plt.plot(ftjj[rindx], ftii[rindx],
                         marker='.', markersize=1,
                         color='b', linestyle='')
                if param_true is not None:
                    ax.plot(param_true[jj], param_true[ii],
                            color='k', marker='s', markersize=4)
                plt.xlim(jjst, jjend)
                plt.ylim(iist, iiend)

            if sciform:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1E'))
                plt.xticks(rotation='vertical')
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1E'))

            if jj == 0 and ii != 0:
                plt.ylabel(r'$%s$' % pname_plt[ii],
                           weight='semibold',
                           fontsize=13)
            else:
                plt.tick_params(
                    axis='y',
                    labelleft=False,
                    labelsize=12)

            if ii == n_par-1:
                plt.xlabel(r'$%s$' % pname_plt[jj],
                           weight='semibold',
                           fontsize=13)
            else:
                plt.tick_params(
                    axis='x',
                    labelbottom=False,
                    labelsize=12)

    if tight_layout:
        plt.tight_layout()

    if pltname is not None:
        plt.savefig(pltname + "_cov.png")
        plt.close()


def plot_hist(flattrace, pname, pname_plt, param_true=None, pltname=None):

    npar = flattrace.shape[1]

    for ii in range(npar):

        plt.figure(figsize=[4, 2])

        figure = sns.distplot(flattrace[:, ii], bins=20)

        if param_true is not None:
            figure.axvline(param_true[ii], color="k")

        plt.xlabel(r'$%s$' % pname_plt[ii], fontsize=15)
        plt.ylabel("relative frequency")

        plt.tight_layout()

        if pltname is not None:
            plt.savefig(pltname + "_hist_" + pname[ii] + ".png")
            plt.close()


def plot_percent_dev(flattrace, feval, D,
                     param_true, xlim,
                     ylim=None, xlabel='x'):

    fig = plt.figure(figsize=[3.25, 1.75])
    ax = fig.add_subplot(111)

    xPlb, xPub = xlim

    nplt = 500
    Tplt = np.linspace(xPlb, xPub, nplt)

    skip = 1
    nlinks_pos = len(flattrace[::skip, :])

    Aclean = np.zeros((nplt, nlinks_pos))

    for ii in range(nplt):
        means = feval(flattrace[::skip], Tplt[ii], D)
        Aclean[ii, :] = means

    pred = np.percentile(Aclean, [50], axis=1)
    true = feval(param_true, Tplt, D)

    dev = 100*(pred-true)/true

    ax.plot(Tplt, np.squeeze(dev), c='b', linestyle='-')

    ax.set_xlim(xPlb, xPub)
    ax.locator_params(nbins=1, axis='x')

    if ylim is None:
        spc = .15
        yPlb = dev.min()-spc*(dev.max()-dev.min())
        yPub = dev.max()+spc*(dev.max()-dev.min())
    else:
        yPlb, yPub = ylim

    plt.ylim(yPlb, yPub)

    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel("% error", fontsize='large')
    plt.tick_params(axis='both', labelsize='large')

    plt.tight_layout()
    plt.tight_layout()


def plot_prediction(flattrace, name_list, Tt, At, It, feval, D,
                    yerr=None, colorL=None,
                    param_true=None, CI=95,
                    xlim=None, ylim=None, pltname=None,
                    xlabel='x', ylabel='y',
                    legend_loc='lower right', Tplt=None,
					ncol=1):

    plt.figure(figsize=[6, 4])

    if colorL is None:
        colorL = sns.color_palette("hls", np.unique(It).size)

    markermat = ['o', 'v', 'p',
                 's', '>', 'P',
                 '*', '<', 'X',
                 'D', 'd', '^']
    if yerr is not None:
        for ii in range(np.unique(It).size):
            plt.errorbar(Tt[It == ii], At[It == ii],
                         yerr=yerr[It == ii],
                         marker=markermat[ii], markersize=5,
                         color=colorL[ii], linestyle='',
                         label=name_list[ii])
    else:
        for ii in range(np.unique(It).size):
            plt.plot(Tt[It == ii], At[It == ii],
                     marker=markermat[ii], markersize=5,
                     color=colorL[ii], linestyle='',
                     label=name_list[ii])

    if xlim is None:
        spc = .15
        xPlb = Tt.min()-spc*(Tt.max()-Tt.min())
        xPub = Tt.max()+spc*(Tt.max()-Tt.min())
    else:
        xPlb, xPub = xlim

    if Tplt is None:
        nplt = 500
        Tplt = np.linspace(xPlb, xPub, nplt)

    skip = 1
    nlinks_pos = len(flattrace[::skip, :])
    Aclean = np.zeros((nplt, nlinks_pos))
    for ii in range(nplt):
        means = feval(flattrace[::skip], Tplt[ii], D)
        Aclean[ii, :] = means

    low, high = np.percentile(Aclean,
                              [0.5*(100-CI), 100-0.5*(100-CI)],
                              axis=1)

    plt.fill_between(Tplt, low, high,
                     alpha=0.3, facecolor='b',
                     label='%s%% CI' % CI)

    plt.plot(Tplt, np.squeeze(feval(np.mean(flattrace, 0), Tplt, D)),
             c='b', label='prediction')

    if param_true is not None:
        plt.plot(Tplt, feval(param_true, Tplt, D),
                 c='k', linestyle=':', label='true model')

    plt.xlim(xPlb, xPub)

    if ylim is None:
        spc = .15
        yPlb = low.min()-spc*(high.max()-low.min())
        yPub = high.max()+spc*(high.max()-low.min())
    else:
        yPlb, yPub = ylim

    plt.ylim(yPlb, yPub)

    plt.legend(loc=legend_loc, shadow=False,
               fontsize=12, ncol=ncol, fancybox=False)
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel(ylabel, fontsize='large')
    plt.tick_params(axis='both', labelsize='large')

    plt.tight_layout()

    if pltname is not None:
        plt.savefig(pltname + "_pred.png")
        plt.close()


def plot_prediction_all(flattrace, name_list, Tt, At, It,
                        pltper,
                        feval_Cp, feval_H,
                        deb_Cp=None,
                        yerr=None, colorLL=None,
                        CI=None,
                        xlim=None, ylim=None, pltname=None,
                        xlabel=None, ylabel=None,
                        legend_loc=None):

    markermat = ['o', 'v', 'p',
                 's', '>', 'P',
                 '*', '<', 'X',
                 'D', 'd', '^']

    Tplt1 = np.linspace(1e-10, 30, 30)[:-1]
    Tplt2 = np.linspace(30, 298.15, 100)[:-1]
    Tplt3 = np.linspace(298.15, 5000, 300)
    Tplt = np.concatenate([Tplt1, Tplt2, Tplt3])
    nplt = Tplt.size

    skip = 1
    flattrace_ = flattrace[::skip, :]

    if deb_Cp is not None:
        deb = np.zeros((flattrace_.shape[0], nplt))
        for ii in range(nplt):
            means = deb_Cp(flattrace_[:, 0], Tplt[ii])
            deb[:, ii] = means

        Cp = feval_Cp(flattrace_, Tplt, deb)
        H = feval_H(flattrace_, Tplt, deb)
    else:
        Cp = feval_Cp(flattrace_, Tplt)
        H = feval_H(flattrace_, Tplt)

    S = cumtrapz(y=Cp/Tplt, x=Tplt, axis=-1, initial=0)
    S -= S[..., 0, None]

    G = H - Tplt*S

    Av = [Cp, H, S, G]

    listcomp = [[ii, jj] for ii in range(len(pltper))
                for jj in range(pltper[ii])]

    for ii, jj in listcomp:

        plt.figure(figsize=[6, 4])

        if colorLL is None:
            colorL = sns.color_palette("hls",
                                       np.unique(It[ii]).size)

        if It[ii] is not None:
            for kk in range(np.unique(It[ii]).size):
                if yerr is not None:
                    plt.errorbar(Tt[ii][It[ii] == kk],
                                 At[ii][It[ii] == kk],
                                 yerr=yerr[ii][It[ii] == kk],
                                 marker=markermat[kk], markersize=5,
                                 color=colorL[kk], linestyle='',
                                 label=name_list[ii][kk])
                else:
                    plt.plot(Tt[ii][It[ii] == kk],
                             At[ii][It[ii] == kk],
                             marker=markermat[kk], markersize=5,
                             color=colorL[kk], linestyle='',
                             label=name_list[ii][kk])

        if CI is None or CI[ii] is None or CI[ii][jj] is None:
            CI_ = 95
        else:
            CI_ = CI[ii][jj]

        low, mid, high = np.percentile(Av[ii],
                                       [0.5*(100-CI_),
                                        50,
                                        100-0.5*(100-CI_)],
                                       axis=0)

        plt.fill_between(Tplt, low, high,
                         alpha=0.3, facecolor='b',
                         label='%s%% CI' % CI_)

        plt.plot(Tplt, mid, c='b', label='prediction')

        if xlim is None or xlim[ii] is None or xlim[ii][jj] is None:
            spc = .15
            xPlb = Tt[ii].min()-spc*(Tt[ii].max()-Tt[ii].min())
            xPub = Tt[ii].max()+spc*(Tt[ii].max()-Tt[ii].min())
        else:
            xPlb, xPub = xlim[ii][jj]
        plt.xlim(xPlb, xPub)

        if ylim is None or ylim[ii] is None or ylim[ii][jj] is None:
            # spacing factor outside of the x limit ranges
            spc = .15

            # restrict range of evaluating y limits to the
            # x range
            low_ = low[(Tplt > xPlb)*(Tplt < xPub)]
            high_ = high[(Tplt > xPlb)*(Tplt < xPub)]

            yPlb = low_.min()-spc*(high_.max()-low_.min())
            yPub = high_.max()+spc*(high_.max()-low_.min())
        else:
            yPlb, yPub = ylim[ii][jj]
        plt.ylim(yPlb, yPub)

        if legend_loc is not None and \
           legend_loc[ii] is not None and \
           legend_loc[ii][jj] is not None:
            plt.legend(loc=legend_loc[ii][jj], shadow=False,
                       fontsize=12, ncol=2, fancybox=False)
        else:
            plt.legend(loc='lower right', shadow=False,
                       fontsize=12, ncol=2, fancybox=False)

        if xlabel is None or xlabel[ii] is None or xlabel[ii][jj] is None:
            plt.xlabel('x', fontsize='large')
        else:
            plt.xlabel(xlabel[ii][jj], fontsize='large')

        if ylabel is None or ylabel[ii] is None or ylabel[ii][jj] is None:
            plt.ylabel('y', fontsize='large')
        else:
            plt.ylabel(ylabel[ii][jj], fontsize='large')

        plt.tick_params(axis='both', labelsize='large')

        plt.tight_layout()

        if pltname is not None and \
           pltname[ii] is not None and \
           pltname[ii][jj] is not None:
            plt.savefig(pltname[ii][jj] + "_pred.png")
            plt.close()


def plot_squiggles(rawtrace, p1, p2, pname_plt, pltname=None):
    nwalkers = rawtrace.shape[0]

    plt.figure(figsize=[4, 3])

    for jj in range(nwalkers):
        plt.plot(rawtrace[jj, 0, p1], rawtrace[jj, 0, p2],
                 'ro')
        plt.plot(rawtrace[jj, :, p1], rawtrace[jj, :, p2],
                 'k-', lw=.5, alpha=.15)

    plt.xlabel(r'$%s$' % pname_plt[p1], fontsize=15)
    plt.ylabel(r'$%s$' % pname_plt[p2], fontsize=15)

    plt.tight_layout()

    if pltname is not None:
        plt.savefig(pltname + "_chain2d.png")
        plt.close()
