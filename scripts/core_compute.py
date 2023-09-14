import sys
import time
import kombine
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
# from emcee import PTSampler
from pymultinest.solve import solve


def coef_summary(flattrace, pname, outname):

    headings = ['parameter', '2.5%ile', '16%ile', '50%ile', '84%ile',
                '97.5%ile', 'Avg.', 'Std. Dev.']

    lvl = np.percentile(flattrace, [2.5, 16, 50, 84, 97.5], axis=0)

    print(flattrace.shape)
    print(lvl.shape)

    S = {}
    S['parameter'] = pname
    S['2.5%ile'] = lvl[0, :]
    S['16%ile'] = lvl[1, :]
    S['50%ile'] = lvl[2, :]
    S['84%ile'] = lvl[3, :]
    S['97.5%ile'] = lvl[4, :]
    S['Avg.'] = np.mean(flattrace, 0)
    S['Std. Dev.'] = np.std(flattrace, 0)

    df = pd.DataFrame(S)
    df = df[headings]

    df.to_csv(outname + '_param.csv')
    print(df.to_string())


def effective_n(mtrace, varnames):
    """this code is taken from pymc3 - N.H. Paulson"""
    """Returns estimate of the effective sample size of a set of traces.
    Parameters
    ----------
    mtrace : MultiTrace or trace object
      A MultiTrace object containing parallel traces (minimum 2)
      of one or more stochastic parameters.
    varnames : list
      Names of variables to include in the effective_n report
    include_transformed : bool
      Flag for reporting automatically transformed variables in addition
      to original variables (defaults to False).
    Returns
    -------
    n_eff : dictionary of floats (MultiTrace) or float (trace object)
        Return the effective sample size, :math:`\hat{n}_{eff}`
    Notes
    -----
    The diagnostic is computed by:
    .. math:: \hat{n}_{eff} = \frac{mn}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}
    where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.
    References
    ----------
    Gelman et al. (2014)"""

    def get_vhat(x):
        # Chain samples are second to last dim (-2)
        num_samples = x.shape[-2]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=-2), axis=-1, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=-2, ddof=1), axis=-1)

        # Estimate marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        return Vhat

    def get_neff(x, Vhat):
        # Number of chains is last dim (-1)
        num_chains = x.shape[-1]

        # Chain samples are second to last dim (-2)
        num_samples = x.shape[-2]

        negative_autocorr = False

        rho = np.ones(num_samples)
        t = 1

        # Iterate until the sum of consecutive estimates of autocorrelation is
        # negative
        while not negative_autocorr and (t < num_samples):

            variogram = np.mean((x[t:, :] - x[:-t, :])**2)
            rho[t] = 1. - variogram / (2. * Vhat)

            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

            t += 1

        if t % 2:
            t -= 1

        return min(num_chains * num_samples,
                   int(num_chains * num_samples / (1. + 2 * rho[1:t-1].sum())))

    def generate_neff(trace_values):
        x = np.array(trace_values)
        shape = x.shape

        # Make sure to handle scalars correctly, adding extra dimensions if
        # needed. We could use np.squeeze here, but we don't want to squeeze
        # out dummy dimensions that a user inputs.
        if len(shape) == 2:
            x = np.atleast_3d(trace_values)

        # Transpose all dimensions, which makes the loop below
        # easier by moving the axes of the variable to the front instead
        # of the chain and sample axes.
        x = x.transpose()

        Vhat = get_vhat(x)

        # Get an array the same shape as the var
        _n_eff = np.zeros(x.shape[:-2])

        # Iterate over tuples of indices of the shape of var
        for tup in np.ndindex(*list(x.shape[:-2])):
            _n_eff[tup] = get_neff(x[tup], Vhat[tup])

        if len(shape) == 2:
            return _n_eff[0]

        return np.transpose(_n_eff)

    n_eff = {}

    for ii in range(len(varnames)):
        # n_eff[var] = generate_neff(mtrace.get_values(var, combine=False))
        n_eff[varnames[ii]] = generate_neff(mtrace[..., ii])

    return n_eff


def gelman_diagnostic(trace, pname):
    """see page 284 of Bayesian Data Analysis, third Ed.
    note: this code assumes that the warm up period has
    already been discarded"""

    nwalkers = trace.shape[0]
    nlinks = trace.shape[1]
    npar = len(pname)

    n_ = np.int32(np.floor(nlinks/2.))  # itterations per segment
    m_ = nwalkers*2  # number of chain segments
    n = np.float32(n_)
    m = np.float32(m_)

    Rhat = {}
    for kk in range(npar):

        psi_i_j = np.zeros((n_, m_))

        psi_i_j[:, :nwalkers] = trace[:, :n_, kk].transpose()
        psi_i_j[:, nwalkers:] = trace[:, n_:, kk].transpose()

        psi_d_j = np.mean(psi_i_j, 0)
        psi_d_d = np.mean(psi_d_j)
        B = (n/(m-1))*np.sum((psi_d_j - psi_d_d)**2)

        s_j_sqrd = (1/(n-1))*np.sum((psi_i_j - psi_d_j)**2, 0)
        W = np.mean(s_j_sqrd)

        mar_pos_var = ((n-1)/n)*W + (1/n)*B

        rnd = np.round(np.sqrt(mar_pos_var/W), 2)

        Rhat[pname[kk]] = "%.2f" % rnd

    return Rhat


def posterior(param, D):
    likelihood = D['likelihood']
    return likelihood(param, D) + prior(param, D)


def sampler_emcee(D):
    # nlinks: number of iterations for each walker
    # please specify as even
    D['nlinks'] = 100

    # nwalkers: number of interacting chains
    D['nwalkers'] = 200

    # ntemps:
    D['ntemps'] = 5

    # ntune: number of initialization steps to discard
    D['ntune'] = 500

    st = time.time()

    # identify starting positions for chains
    tmp = start_pos(D['ntemps']*D['nwalkers'], D)
    pos0 = tmp.reshape((D['ntemps'], D['nwalkers'], D['dim']))

    likelihood = D['likelihood']

    # run MCMC for the model of interest
    sampler = PTSampler(ntemps=D['ntemps'], nwalkers=D['nwalkers'],
                        dim=len(D['pname']),
                        logl=likelihood, logp=prior,
                        loglargs=(D,),
                        logpargs=(D,))

    for pos, lnprob, lnlike in sampler.sample(pos0, iterations=D['ntune']):
        pass
    burntrace = sampler.chain
    sampler.reset()

    print(burntrace.shape)
    print("burnin completed")

    for pos, lnprob, lnlike in sampler.sample(pos, iterations=D['nlinks']):
        pass
    finaltrace = sampler.chain
    print(finaltrace.shape)

    D['rawtrace'] = np.concatenate((burntrace[0, ...], finaltrace[0, ...]),
                                   axis=1)

    D['sampling_time'] = np.round(time.time()-st, 2)

    st = time.time()

    D['lnZ'], D['dlnZ'] = sampler.thermodynamic_integration_log_evidence()

    elaps = np.round(time.time()-st, 2)
    msg = "model evidence evaluation: " + str(elaps) + " seconds"
    WP(msg, D['wrt_file'])

    return D


def sampler_kombine(D):

    st = time.time()

    # nlinks: number of iterations for each walker
    # please specify as even
    D['nlinks'] = 50

    # nwalkers: number of interacting chains
    D['nwalkers'] = 400

    # identify starting positions for chains
    pos = start_pos(D['nwalkers'], D)

    # run MCMC for the model of interest
    sampler = kombine.sampler.Sampler(D['nwalkers'], len(D['pname']),
                                      posterior,
                                      args=(D,))

    sampler.burnin(p0=pos)
    burntrace = sampler.chain[...].swapaxes(0, 1)

    print(burntrace.shape)
    print("burnin completed")

    sampler.run_mcmc(N=D['nlinks'])

    D['sampling_time'] = np.round(time.time()-st, 2)

    D['lnZ'], D['dlnZ'] = sampler.ln_ev(np.int32(1e3))

    D['autocorr_times'] = sampler.autocorrelation_times
    au_mu = np.mean(D['autocorr_times'])
    au_sig = np.std(D['autocorr_times'])
    msg = "mean, std of autocorrelation times: " + \
          str(au_mu) + ", " + str(au_sig)
    WP(msg, D['wrt_file'])

    D['accept_frac'] = sampler.acceptance_fraction
    ar_mu = np.mean(D['accept_frac'])
    ar_sig = np.std(D['accept_frac'])
    msg = "mean, std of acceptance fractions: " + \
        str(ar_mu) + ", " + str(ar_sig)
    WP(msg, D['wrt_file'])

    D['rawtrace'] = sampler.chain[...].swapaxes(0, 1)

    return D


def sampler_multinest(D):

    st = time.time()

    if not os.path.exists("chains"):
        os.mkdir("chains")

    prefix = "chains/" + D['outname'] + "-"

    # number of live points to maintain in the multinest sampler
    D['nlive'] = 800

    def L(param):
        likelihood = D['likelihood']
        return likelihood(param, D)

    def P(cube): return prior_multinest(cube, D)

    result = solve(LogLikelihood=L, Prior=P,
                   n_dims=D['nparam'],
                   n_live_points=D['nlive'],
                   outputfiles_basename=prefix,
                   sampling_efficiency=0.8,
                   evidence_tolerance=0.5,
                   resume=False)

    D['lnZ'] = result['logZ']
    D['dlnZ'] = result['logZerr']
    D['rawtrace'] = result['samples']
    print(D['rawtrace'].shape)

    D['sampling_time'] = np.round(time.time()-st, 2)

    return D


def prior(param, D):
    """
    compute the prior probability density for a selected
    parameter set. Note that this is expressed as a log
    probability density to match with our log likelihood
    """

    prior_sum = 0

    for ii in range(D['dim']):
        if D['distV'][ii] == 'uniform':
            prior_sum += ss.uniform.logpdf(param[ii],
                                           loc=D['locV'][ii],
                                           scale=D['scaleV'][ii])
        elif D['distV'][ii] == 'norm':
            prior_sum += ss.norm.logpdf(param[ii],
                                        loc=D['locV'][ii],
                                        scale=D['scaleV'][ii])
        elif D['distV'][ii] == 'expon':
            prior_sum += ss.expon.logpdf(param[ii])
        elif D['distV'][ii] == 'triang':
            prior_sum += ss.triang.logpdf(param[ii],
                                          c=D['cV'][ii],
                                          loc=D['locV'][ii],
                                          scale=D['scaleV'][ii])
        else:
            sys.exit("invalid distribution type selected")

    return prior_sum


def prior_multinest(cube, D):
    """
    convert the unit cube into the parameter cube
    see:
    https://johannesbuchner.github.io/PyMultiNest/pymultinest_run.html
    """
    cube_p = np.zeros(cube.shape)

    for ii in range(D['dim']):
        if D['distV'][ii] == 'uniform':
            cube_p[ii] = ss.uniform.ppf(cube[ii],
                                        loc=D['locV'][ii],
                                        scale=D['scaleV'][ii])
        elif D['distV'][ii] == 'norm':
            cube_p[ii] = ss.norm.ppf(cube[ii],
                                     loc=D['locV'][ii],
                                     scale=D['scaleV'][ii])
        elif D['distV'][ii] == 'expon':
            cube_p[ii] = ss.expon.ppf(cube[ii])
        elif D['distV'][ii] == 'triang':
            cube_p[ii] = ss.triang.ppf(cube[ii],
                                       c=D['cV'][ii],
                                       loc=D['locV'][ii],
                                       scale=D['scaleV'][ii])
        elif D['distV'][ii] == 'custom':
            custom = D['custom']
            cube_p[ii] = custom(cube, cube_p, D)
        else:
            sys.exit("invalid distribution type selected")

    return cube_p


def start_pos(size, D):

    pos = np.zeros((size, D['dim']))

    for ii in range(D['dim']):
        if D['distV'][ii] == 'uniform':
            pos[:, ii] = ss.uniform.rvs(loc=D['locV'][ii],
                                        scale=D['scaleV'][ii],
                                        size=size)
        elif D['distV'][ii] == 'norm':
            pos[:, ii] = ss.norm.rvs(loc=D['locV'][ii],
                                     scale=D['scaleV'][ii],
                                     size=size)
        elif D['distV'][ii] == 'expon':
            pos[:, ii] = ss.expon.rvs(size=size)
        elif D['distV'][ii] == 'triang':
            pos[:, ii] = ss.triang.rvs(c=D['cV'][ii],
                                       loc=D['locV'][ii],
                                       scale=D['scaleV'][ii],
                                       size=size)
        else:
            sys.exit("invalid distribution type selected")

    return pos


def WP(msg, filename):
    """
    Summary:
        This function takes an input message and a filename, and appends that
        message to the file. This function also prints the message
    Inputs:
        msg (string): the message to write and print.
        filename (string): the full name of the file to append to.
    Outputs:
        both prints the message and writes the message to the specified file
    """
    fil = open(filename, 'a')
    print(msg)
    fil.write(msg)
    fil.write('\n')
    fil.close()


if __name__ == '__main__':

    print("do something'")
