#!/usr/bin/env python3

"""a one-off script to sample a more complicated structure function ansatz for a larger dataset
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

import h5py

import jax
from jax import random
from jax import numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (MCMC, NUTS, init_to_value)
from numpyro.diagnostics import effective_sample_size

numpyro.enable_x64() # improve default numerical precision

from w4t.utils.infer import (thin, scaling_exponent_ansatz, _sample_sea_xcb_prior)

#-------------------------------------------------

DEFAULT_NUM_WARMUP = 1000
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_RETAINED = np.inf

DEFAULT_SEED = 123

#-------------------------------------------------

def load_structure_function_dat(
        paths,
        index,
        long_or_trsv,
        min_rtol=0.0,
        min_atol=0.0,
        min_scale=0.0,
        max_scale=0.5,
        logspace_scales=False,
        verbose=False,
    ):
    scales = None
    mom = []

    tmp = 'SF%sorder%02d' % (long_or_trsv, index)

    for path in paths:
        if verbose:
            print('loading structure function estimates for (%s, order=%d) from: %s' & \
                (long_or_trsv, index, path))

        data = np.genfromtxt(path, names=True)
        if scales is None:
            scales = data['01_GridStag']
        else:
            assert np.all(scales == data['01_GridStag']), 'mismatch in increments'
        key = [key for key in data.dtype.names if (tmp in key)]
        assert len(key) == 1
        key = key[0]

        mom.append(data[key])

    # compute stdv
    if len(paths) > 1:
        stdv = np.std(mom, axis=0) / len(paths)**0.5 # stdv between files scaled by the number of files
        mom = np.mean(mom, axis=0)
    else:
        mom = mom[0]
        stdv = np.zeros_like(scales, dtype=float)

    if min_rtol > 0: # set a lower limit on the relative uncertainty
        if verbose:
            print('limitting stdv to rtol >= %.3e' % min_rtol)
        stdv = np.where(stdv < min_rtol*mom, min_rtol*mom, stdv)

    if min_atol > 0:
        if verbose:
            print('limitting stdv to atol >= %.3e' % min_atol)
        stdv = np.where(stdv < min_atol, min_atol, stdv)

    # interpolate to get fewer data
    if logspace_scales is not None:
        if verbose:
            print('resampling %d linearly-spaced points to %d logarithmically spaced points' % \
                (len(scales), logspace_scales))

        new_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), logspace_scales)
        mom = np.interp(new_scales, scales, mom)
        stdv = np.interp(new_scales, scales, stdv)
        scales = new_scales

    else:
        if verbose:
            print('downselecting to scales between %.3e - %.3e' % (min_scale, max_scale))
        sel = (min_scale <= scales) * (scales < max_scale)
        scales = scales[sel]
        mom = mom[sel]
        stdv = stdv[sel]

    # return
    return scales, mom, stdv

#------------------------

def write_structure_function_samples(
        path,
        posterior,
        prior,
        data,
        verbose=False,
        **meta
    ):
    if verbose:
        print('writing samples to: '+path)

    with h5py.File(path, 'w') as obj:
        for k, v in meta.items():
            obj.attrs.create(k, v)

        for grp, data in [
                (obj.create_group('posterior'), posterior),
                (obj.create_group('prior'), prior),
                (obj.create_group('data'), data),
            ]:
            for k, v in data.items():
                grp.create_dataset(k, data=v)

#---

def load_structure_function_samples(path, verbose=False):
    if verbose:
        print('loading structure function samples from: '+path)

    with h5py.File(path, 'r') as obj:
        posterior = dict((k, v[:]) for k, v in obj['posterior'].items())
        prior = dict((k, v[:]) for k, v in obj['prior'].items())
        data = dict((k, v[:]) for k, v in obj['data'].items())
        meta = dict(obj.attrs.items())

    return posterior, prior, data, meta

#------------------------

def write_scaling_exponent_samples(
        path,
        posterior,
        prior,
        verbose=False,
        **meta
    ):
    if verbose:
        print('writing samples to: '+path)

    with h5py.File(path, 'w') as obj:
        for k, v in meta.items():
            obj.attrs.create(k, v)

        for grp, data in [
                (obj.create_group('posterior'), posterior),
                (obj.create_group('prior'), prior),
            ]:
            for k, v in data.items():
                grp.create_dataset(k, data=v)

#---

def load_scaling_exponent_samples(path, verbose=False):
    if verbose:
        print('loading structure function samples from: '+path)

    with h5py.File(path, 'r') as obj:
        posterior = dict((k, v[:]) for k, v in obj['posterior'].items())
        prior = dict((k, v[:]) for k, v in obj['prior'].items())
        meta = dict(obj.attrs.items())

    return posterior, prior, meta

#-------------------------------------------------

def structure_function_ansatz(scales, amp, xi, s1, b1, n1, s2, b2, n2, s3, b3, n3, ref_scale=0.5):
    return amp * (scales/ref_scale)**xi \
        * (1 + (scales/s1)**n1)**(b1/n1) \
        * (1 + (scales/s2)**n2)**(b2/n2) \
        * (1 + (scales/s3)**n3)**(b3/n3)

def logarithmic_derivative_structure_function_ansatz(scales, amp, xi, s1, b1, n1, s2, b2, n2, s3, b3, n3, ref_scale=0.5):
    return xi + b1/(1+(scales/s1)**(-n1)) + b2/(1+(scales/s2)**(-n2)) + b3/(1+(scales/s3)**(-n3))

def averaged_logarithmic_derivative_ansatz(low, high, *args):
    return np.log(structure_function_ansatz(high, *args) / structure_function_ansatz(low, *args)) / np.log(high/low)

#------------------------

def _sample_sfa_prior(
        mean_logamp=-10.0,
        stdv_logamp=10.0,
        mean_xi=0.0,
        stdv_xi=3.0,
        mean_logs1=np.log(1e-3),
        stdv_logs1=1.0, 
        min_b1=-10.0,
        max_b1=+0.0,
        min_n1=+0.0,
        max_n1=+5.0,
        mean_logs2=np.log(4e-2),
        stdv_logs2=1.0,
        min_b2=-0.0,
        max_b2=+10.0,
        min_n2=+0.0, 
        max_n2=+5.0,
        mean_logs3=np.log(3e-1),
        stdv_logs3=1.0,
        min_b3=-10.0,
        max_b3=+0.0,
        min_n3=+0.0,
        max_n3=+5.0,
        **ignored
    ):  
    amp = _sample_sfa_amp_prior(mean_logamp=mean_logamp, stdv_logamp=stdv_logamp)
                
    xi = _sample_sfa_xi_prior(mean_xi=mean_xi, stdv_xi=stdv_xi)
    
    s1, b1, n1 = _sample_sfa_sbn_prior(
        mean_logs=mean_logs1,
        stdv_logs=stdv_logs1,
        min_b=min_b1,
        max_b=max_b1,
        min_n=min_n1,
        max_n=max_n1,
        suffix='1',
    )

    s2, b2, n2 = _sample_sfa_sbn_prior(
        mean_logs=mean_logs2,
        stdv_logs=stdv_logs2,
        min_b=min_b2,
        max_b=max_b2,
        min_n=min_n2,
        max_n=max_n2,
        suffix='2',
    )

    s3, b3, n3 = _sample_sfa_sbn_prior(
        mean_logs=mean_logs3,
        stdv_logs=stdv_logs3,
        min_b=min_b3,
        max_b=max_b3,
        min_n=min_n3,
        max_n=max_n3,
        suffix='3',
    )

    return amp, xi, s1, b1, n1, s2, b2, n2, s3, b3, n3

def _sample_sfa_amp_prior(mean_logamp=-10.0, stdv_logamp=10.0):
    return numpyro.sample("amp", dist.LogNormal(mean_logamp, stdv_logamp))

def _sample_sfa_xi_prior(mean_xi=0.0, stdv_xi=3.0):
    return numpyro.sample("xi", dist.Normal(mean_xi, stdv_xi))

def _sample_sfa_sbn_prior(
        mean_logs=np.log(10),
        stdv_logs=1.0,
        min_b=-3.0,
        max_b=+3.0,
        min_n=0.0,
        max_n=5.0,
        suffix='',
    ):
    s = numpyro.sample("s"+suffix, dist.LogNormal(mean_logs, stdv_logs))
    b = numpyro.sample("b"+suffix, dist.Uniform(min_b, max_b))
    n = numpyro.sample("n"+suffix, dist.Uniform(min_n, max_n))
    return s, b, n

#------------------------

def sample_structure_function_ansatz(
        scales,
        mom,
        std,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        num_retained=DEFAULT_NUM_RETAINED,
        seed=[DEFAULT_SEED],
        verbose=False,
        num_segs=1,
        ref_scale=0.5,
        **prior_kwargs
    ):
    """sample for parameters of a simple model for structure function scaling
    """
    if numpyro is None:
        raise ImportError('could not import numpyro')

    if verbose:
        print('defining model')

    def sample_posterior(obs):
        # draw from prior
        params = _sample_sfa_prior(**prior_kwargs)

        # compute expected value
        sf = structure_function_ansatz(scales, *params, ref_scale=ref_scale)

        # compare to observed data
        numpyro.sample('mom', dist.Normal(sf, std), obs=obs)

    #---

    # run the sampler

    Prior = None

    for s in seed:

        if verbose:
            print('running sampler for prior with seed=%d for %d warmup and %d samples' % (s, num_warmup, num_samples))

        try:
            mcmc = MCMC(NUTS(_sample_sfa_prior), num_warmup=num_warmup, num_samples=num_samples)
            mcmc.run(random.PRNGKey(s), **prior_kwargs)
        except Exception as e:
            if verbose:
                print('>>> sampler failed!')
                print(e)
            continue

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        prior = mcmc.get_samples()
        prior = thin(num_samples, prior, prior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            prior = dict((key, val[-num_retained:]) for key, val in prior.items())

        if Prior is None:
            Prior = dict((k, [v]) for k, v in prior.items())

        else:
            for k, v in prior.items():
                Prior[k].append(v)
    assert Prior is not None, 'no seeds succeeded for the prior!'

    #---

    Posterior = None

    for s in seed:
        if verbose:
            print('running sampler for posterior with seed=%d for %d warmup and %d samples' % \
                (s, num_warmup, num_samples))

        try:
            mcmc = MCMC(NUTS(sample_posterior), num_warmup=num_warmup, num_samples=num_samples)
            mcmc.run(random.PRNGKey(s), mom)
        except Exception as e:
            if verbose:
                print('>>> sampler failed!')
                print(e)
            continue

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        posterior = mcmc.get_samples()
        posterior = thin(num_samples, posterior, posterior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            posterior = dict((key, val[-num_retained:]) for key, val in posterior.items())

        # record the likelihood of each sample

        if verbose:
            print('computing likelihood at samples')

        posterior.update(numpyro.infer.log_likelihood(sample_posterior, posterior, mom))

        if Posterior is None:
            Posterior = dict((k, [v]) for k, v in posterior.items())

        else:
            for k, v in posterior.items():
                Posterior[k].append(v)

    assert Posterior is not None, 'no seeds succeeded for the posterior!'

    #---

    Posterior = dict((k, np.concatenate(tuple(v))) for k, v in Posterior.items())
    Prior = dict((k, np.concatenate(tuple(v))) for k, v in Prior.items())

    if verbose:
        print('\n>>> retained a total of %d prior samples' % len(list(Prior.values())[0]))
        print('>>> retained a total of %d posterior samples\n' % len(list(Posterior.values())[0]))

    # return
    return Posterior, Prior

#-------------------------------------------------

def simple_sample_scaling_exponent_ansatz(
        indexes,
        samples,
        ref_scale,
        num_warmup=DEFAULT_NUM_WARMUP,
        num_samples=DEFAULT_NUM_SAMPLES,
        num_retained=DEFAULT_NUM_RETAINED,
        seed=[DEFAULT_SEED],
        verbose=False,
        num_segs=1,
        xi_prior_mean=0.0,
        xi_prior_stdv=3.0,
        **prior_kwargs
    ):
    """sample the distribution of scaling exponents using a KDE model of marginal likelihoods
    """
    if verbose:
        print('defining model')

    num_indexes = len(indexes)
    
    # computing KDE models of averaged scaling exponents
    grids = []
    kdes = []

    for ind, i in enumerate(indexes):
        if verbose:
            print('    computing KDE for xi(index=%d) with %d samples' % (i, len(samples[i]['amp'])))
    
        samp = averaged_logarithmic_derivative_ansatz(
            ref_scale[0],
            ref_scale[1],
            samples[i]['amp'],
            samples[i]['xi'],
            samples[i]['sl'],
            samples[i]['bl'],
            samples[i]['nl'],
            samples[i]['sh'],
            samples[i]['bh'],
            samples[i]['nh'],
        )

        # NOTE! this assumes a default prior, which could be fragile
        # prior for xi that we're going to undo
        # in practice, xi is constrained well compared to the mean so that weights are all about the same
        w = 1./(np.exp(0.5*(samples[i]['xi']-xi_prior_mean)**2/3**2) / (2*np.pi * xi_prior_stdv**2)**0.5)
        w /= np.sum(w)

        ### simple Gaussian approximation
        m = np.sum(w*samp)             # sample mean
        s = np.sum(w*(samp-m)**2)**0.5 # sample stdv (sorta)

        ### KDE of samples
        kde_stdv = 1.06 * s / len(samp)**0.2 # rule of thumb for optimal KDE bandwidth w/r/t IMSE

        grid = np.min(samp), np.max(samp)
        d = 0.5*(grid[1]-grid[0])
        grid = np.linspace(grid[0]-d, grid[1]+d, 101)
        kde = np.sum(
            [W*np.exp(-0.5*(S-grid)**2/kde_stdv**2)/(2*np.pi*kde_stdv**2)**0.5 for W, S in zip(w, samp)],
            axis=0,
        )

        grids.append(grid)
        kdes.append(kde)

    grids = np.array(grids, dtype=float)
    kdes = np.array(kdes, dtype=float)

    # finish defining model

    def sample_posterior(**prior_kwargs):
        x, C0, beta = _sample_sea_xcb_prior(**prior_kwargs)                 # sample from prior
        xi = scaling_exponent_ansatz(indexes, x, C0, beta)  # compute means
        for ind, i in enumerate(indexes): # this shouldn't be too bad, but may need to vmap
            numpyro.factor('data_%d'%i, jnp.log(jnp.interp(xi[ind], grids[ind], kdes[ind])))

    #---

    Prior = None

    for s in seed:

        # run the sampler

        if verbose:
            print('running sampler for prior with seed=%d for %d warmup and %d samples' % \
                (s, num_warmup, num_samples))

        try:
            mcmc = MCMC(
                NUTS(_sample_sea_xcb_prior, init_strategy=init_to_value(values=init_xcb_values)),
                num_warmup=num_warmup,
                num_samples=num_samples,
            )
            mcmc.run(random.PRNGKey(s), **prior_kwargs)
        except:
            if verbose:
                print('>>> sampler failed!')
            continue

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        prior = mcmc.get_samples()
        prior = thin(num_samples, prior, prior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            prior = dict((key, val[-num_retained:]) for key, val in prior.items())

        if Prior is None:
            Prior = dict((k, [v]) for k, v in prior.items())

        else:
            for k, v in prior.items():
                Prior[k].append(v)

    assert Prior is not None, 'no seeds succeeded for the prior!'

    #---

    Posterior = None

    for s in seed:

        if verbose:
            print('running sampler for posterior with seed=%d for %d warmup and %d samples' % \
                (s, num_warmup, num_samples))

        try:
            mcmc = MCMC(
                NUTS(sample_posterior, init_strategy=init_to_value(values=init_xcb_values)),
                num_warmup=num_warmup,
                num_samples=num_samples,
            )
            mcmc.run(random.PRNGKey(s), **prior_kwargs)
        except:
            if verbose:
                print('>>> sampler failed!')
            continue

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

        posterior = mcmc.get_samples()
        posterior = thin(num_samples, posterior, posterior.keys(), num_segs=num_segs, verbose=verbose)

        if num_retained < np.inf:
            if verbose:
                print('retaining the final %d samples' % num_retained)
            posterior = dict((key, val[-num_retained:]) for key, val in posterior.items())

        # record the likelihood of each sample
        if verbose:
            print('computing likelihood at samples')

        posterior.update(numpyro.infer.log_likelihood(sample_posterior, posterior, **prior_kwargs))

        if Posterior is None:
            Posterior = dict((k, [v]) for k, v in posterior.items())

        else:
            for k, v in posterior.items():
                Posterior[k].append(v)

    assert Posterior is not None, 'no seeds succeeded for the posterior!'

    #---

    Posterior = dict((k, np.concatenate(tuple(v))) for k, v in Posterior.items())
    Prior = dict((k, np.concatenate(tuple(v))) for k, v in Prior.items())

    if verbose:
        print('\n>>> retained a total of %d prior samples' % len(list(Prior.values())[0]))
        print('>>> retained a total of %d posterior samples\n' % len(list(Posterior.values())[0]))

    # return
    return Posterior, Prior
