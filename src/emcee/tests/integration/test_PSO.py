# -*- coding: utf-8 -*-

import pytest
import numpy as np
import emcee

from emcee import moves

__all__ = [
    "test_normal_pso",
    "test_uniform_pso"
]

def rosenbrock( z ): 
    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
    z = np.asarray_chkfinite(z)
    x = z[0]
    y = z[1]

    f = (1. - x)**2  + 100.*(y - x**2)**2
    return -np.log(f)

def normal_log_prob_blobs(params):
    return -0.5 * np.sum(params ** 2), params

def normal_log_prob(params):
    return -0.5 * np.sum(params ** 2)

def _test_normal(
    proposal,
    ndim=1,
    nwalkers=32,
    nsteps=2000,
    seed=1234,
    check_acceptance=True,
    pool=None,
    blobs=False,
):
    # Set up the random number generator.
    np.random.seed(seed)

    # Initialize the ensemble and proposal.
    coords = np.random.randn(nwalkers, ndim)

    if blobs:
        lp = normal_log_prob_blobs
    else:
        lp = normal_log_prob

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lp, moves=proposal, pool=pool
    )
    if hasattr(proposal, "ntune") and proposal.ntune > 0:
        coords = sampler.run_mcmc(coords, proposal.ntune, tune=True)
        sampler.reset()
    sampler.run_mcmc(coords, nsteps)

    # Check the acceptance fraction.
    if check_acceptance:
        acc = sampler.acceptance_fraction
        assert np.all(
            (acc <= 1.0) * (acc >= 0.0)
        ), "Invalid acceptance fraction\n{0}".format(acc)

    # Check that the MAP is within tolerance
    logps = sampler.get_log_prob(flat=True)
    samps = sampler.get_chain(flat=True)

    idx_map = logps.argmax()
    mu_map  = samps[idx_map]
    assert np.all(np.abs(mu_map) < 0.08), "Incorrect mean"


def _test_rosenbrock(
    proposal,
    ndim=2,
    nwalkers=50,
    nsteps=2000,
    seed=1234):
    # Set up the random number generator.
    np.random.seed(seed)

    # Initialize the ensemble and proposal.
    coords = np.random.randn(nwalkers, ndim)


    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, rosenbrock, moves=proposal)

    sampler.run_mcmc(coords, nsteps)

    # Check that the MAP is within tolerance
    logps = sampler.get_log_prob(flat=True)
    samps = sampler.get_chain(flat=True)

    idx_map = logps.argmax()
    mu_map  = samps[idx_map]
    assert np.all(np.abs(mu_map - [1.,1.]) < 1e-3), "Incorrect solution"


@pytest.mark.parametrize("blobs", [True,False])
def test_normal_pso(blobs, **kwargs):
    kwargs["blobs"] = blobs
    _test_normal(moves.PSOMove(), **kwargs)

def test_rosenbrock( **kwargs):
    _test_rosenbrock(moves.PSOMove(), **kwargs)
