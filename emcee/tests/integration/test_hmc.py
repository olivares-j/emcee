# -*- coding: utf-8 -*-

from __future__ import division, print_function

from multiprocessing import Pool
import pytest
import numpy as np

from emcee import moves
from emcee.moves.hmc import (
    IdentityMetric, IsotropicMetric, DiagonalMetric, DenseMetric
)

from .test_proposal import _test_normal

pools = [False]
if hasattr(Pool, "__exit__"):
    pools = [True, False]


@pytest.mark.parametrize("metric", [None, IdentityMetric(3),
                                    IsotropicMetric(3),
                                    DiagonalMetric(np.ones(3)),
                                    DenseMetric(np.eye(3))])
@pytest.mark.parametrize("pool", pools)
@pytest.mark.parametrize("blobs", [True, False])
def test_normal_hmc(pool, blobs, metric, **kwargs):
    move = moves.HamiltonianMove(5, metric=metric)
    kwargs["tune"] = 1200
    kwargs["ndim"] = 3
    kwargs["nwalkers"] = 2
    kwargs["check_acceptance"] = False
    kwargs["nsteps"] = 1500
    kwargs["blobs"] = blobs
    if pool:
        with Pool() as p:
            kwargs["pool"] = p
            _test_normal(move, **kwargs)
    else:
        _test_normal(move, **kwargs)