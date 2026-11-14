"""Contract for pravah.experiments.robust_design (E15; docs/research/robust-placement-design.md)."""
import dataclasses

import numpy as np
import pytest

from pravah import benchmarks as B
from pravah import placement as P
from pravah.experiment_config import ExperimentConfig, Split, Winds
from pravah.experiments import cross_objective as X
from pravah.experiments import forecast_design as FD
from pravah.experiments import robust_design as RD


# ---------------------------------------------------------------- risk measures
def test_cvar_ordering_and_rockafellar_uryasev():
    rng = np.random.default_rng(0)
    for n in (5, 20, 44):
        L = rng.gamma(2, 1, n)
        for a in (0.0, 0.5, 0.8, 0.95):
            c = RD.cvar(L, a)
            eta = np.concatenate([L, [L.min() - 1]])  # the RU minimum over eta is attained at a sample point
            ru = min(e + np.maximum(L - e, 0).sum() / ((1 - a) * n) for e in eta)
            assert np.isclose(c, ru, rtol=1e-12)
            assert L.mean() - 1e-12 <= c <= L.max() + 1e-12
        assert np.isclose(RD.cvar(L, 0.0), L.mean())
        assert np.isclose(RD.cvar(L, 1 - 1 / n), L.max())  # tail of exactly one scenario
        assert RD.cvar(L, .5) <= RD.cvar(L, .8) + 1e-12 <= RD.cvar(L, .9) + 2e-12  # monotone in alpha


def test_ensembles_exclude_test_directions():
    for name, blocks in RD.ENSEMBLES.items():
        dirs = {d for kind, d, _ in blocks} | {(d + 1) % FD.N_DIRS for kind, d, _ in blocks if kind == "veer"}
        assert not dirs & set(RD.TEST_DIRS), name
    assert {d for _, d, _ in RD.ENSEMBLES["train3"]} == {2, 6, 7}  # E, W, NW: the training regimes' directions


# ---------------------------------------------------------------- synthetic tiny problem: objective vs brute force
def _synthetic(n_sites=12, K=2, n_blocks=3, T=30, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, (n_sites, 2))
    p = P.Problem(xy=xy, cand=np.arange(n_sites), min_sep=0.0, a_ref=1.0, sigma=1.0,
                  G3=rng.gamma(1, 1, (n_sites, T, K)) * (rng.random((n_sites, 1, K)) < .5))
    Fb = rng.gamma(1, 1, (K, n_blocks * FD.BLOCK_H, 6))
    G = rng.gamma(1, .5, (n_sites, n_blocks * FD.BLOCK_H, K)) * (rng.random((n_sites, n_blocks, 1, K)) < .4).repeat(FD.BLOCK_H, 2).reshape(n_sites, -1, K)
    Q = np.einsum("khn,lhn->hkl", Fb, Fb).reshape(n_blocks, FD.BLOCK_H, K, K).mean(1) / 6
    m2 = np.stack([np.eye(K)] * (n_blocks - 1) + [FD.loguniform_m2(.5, 5, K)])
    return p, RD.Ensemble(G, Q, m2, [("s", i, 3.0, "x") for i in range(n_blocks)])


def _explicit(p, ens, S, memory):
    K = ens.Q.shape[1]
    out = []
    for s in range(len(ens.Q)):
        A = ens.G[S][:, s * FD.BLOCK_H:(s + 1) * FD.BLOCK_H]
        M = sum(A[i].T @ A[i] for i in range(len(S)))
        if memory == "full":
            M = M + sum(p.G3[i].T @ p.G3[i] for i in S)
        Sig = np.linalg.inv(np.eye(K) + M)
        out.append(np.trace(ens.Q[s] @ Sig @ (ens.m2[s] + M) @ Sig) / np.trace(ens.Q[s] @ ens.m2[s]))
    return np.array(out)


@pytest.mark.parametrize("memory", RD.MEMORIES)
def test_losses_match_explicit_and_greedy_reaches_brute_force(memory):
    p, ens = _synthetic()
    obj = RD.RobustObjective(p, ens, memory)
    S = [1, 4, 7]
    assert np.allclose(obj.losses(S), _explicit(p, ens, S, memory), rtol=1e-12)
    assert np.allclose(obj.losses([]), 1.0)  # empty network: the normalising loss
    for c in RD.CRITERIA:
        v = obj.value(c)
        best, bS, _ = P.brute_force(p, 2, v)
        assert np.isclose(-best, RD.AGG[c](_explicit(p, ens, bS, memory)))
        S2, info = RD.make_robust_design(p, obj, c, 2, {"s0": [0, 1]})
        assert np.isclose(v(p, S2), best, rtol=1e-10), (memory, c)


def test_loss_monotone_in_sites_for_unit_prior():
    p, ens = _synthetic(n_blocks=2)
    ens = RD.Ensemble(ens.G, ens.Q, np.stack([np.eye(2)] * 2), ens.labels)
    obj = RD.RobustObjective(p, ens, "w24")
    S = []
    for i in range(6):
        L0 = obj.losses(S)
        S.append(i)
        assert (obj.losses(S) <= L0 + 1e-12).all()


# ---------------------------------------------------------------- real tiny world: blocks and leakage
@pytest.fixture(scope="module")
def tiny():
    c = ExperimentConfig.default("EXP-E15-test")
    c.model.grid, c.model.spinup_h = (16, 16), 6
    c.design.n_sources = (2,)
    c.split = Split(train=(1, 2), validation=(3, 3), test=(4, 5))
    c.winds = Winds(train=["veering"], heldout=c.winds.heldout)
    sc = B.generate_scenario(c, 0, 2)
    F, G3 = X.responses(sc)
    lab = [("steady", 2, 1.5), ("steady", 6, 3.0), ("veer", 6, 3.0)]
    blocks = [(b, np.stack([RD.block_field(c, xy, b, 6) for xy in sc.truth.source_xy])) for b in lab]
    return sc, F, G3, blocks


def test_steady_block_equals_e11_prior_block(tiny):
    sc, _, _, blocks = tiny
    assert np.array_equal(blocks[1][1][0], FD.prior_block(sc.cfg, sc.truth.source_xy[0], 6, 6))
    w = RD.block_winds("veer", 6, 3.0)
    assert np.allclose(w[:12], FD.design_winds()[6 * 24:6 * 24 + 12]) and np.allclose(w[12:], FD.design_winds()[7 * 24:7 * 24 + 12])


def _robust_designs(sc, G3, blocks, k=3):
    di = X.design_input(sc, G3, X.site_sigma(sc, False))
    p = X.problem(di)
    ens = RD.Ensemble.build(blocks, lambda F: X.site_responses(sc, F), di.a_ref, FD.loguniform_m2(*di.prior_loguniform, 2))
    return {(m, c): RD.make_robust_design(p, RD.RobustObjective(p, ens, m), c, k)[0] for m in RD.MEMORIES for c in RD.CRITERIA}


def test_designs_unchanged_when_test_info_is_nan(tiny):
    sc, F, G3, blocks = tiny
    ref = _robust_designs(sc, G3, blocks)
    T = len(sc.split_hours("train"))
    G3p = G3.copy()
    G3p[:, T:] = np.nan  # validation/test-hour sensitivities
    nan = lambda a: np.full_like(np.asarray(a, float), np.nan)
    sc2 = B.Scenario(sc.cfg, sc.replicate, sc.n_sources,
                     dataclasses.replace(sc.truth, field=nan(sc.truth.field), strengths_ug_s=nan(sc.truth.strengths_ug_s),
                                         site_truth=nan(sc.truth.site_truth), winds=nan(sc.truth.winds)),
                     sc.site_ij, sc.site_xy, sc.noise_sigma, {k: nan(v) for k, v in sc._noise.items()}, sc.lattice_n)
    sig = X.site_sigma(sc, False)  # protocol sigma (training truth), passed as in the E11 test
    di = X.design_input(sc2, G3p, sig)
    p = X.problem(di)
    ens = RD.Ensemble.build(blocks, lambda Fb: X.site_responses(sc2, Fb), di.a_ref, FD.loguniform_m2(*di.prior_loguniform, 2))
    got = {(m, c): RD.make_robust_design(p, RD.RobustObjective(p, ens, m), c, 3)[0] for m in RD.MEMORIES for c in RD.CRITERIA}
    assert got == ref
    assert all(np.isfinite(RD.RobustObjective(p, ens, m).losses(S)).all() for (m, _), S in got.items())
