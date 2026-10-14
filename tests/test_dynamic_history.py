import json

import numpy as np
import pytest

from pravah import dynamic as dyn
from pravah.studies.toy import dynamic_history as dh
from pravah.studies.toy.dynamic_study import DUR, N_FINE, N_INV, NP, REF, _hash

PER = 12  # observations per bin


def _problem(seed=0, upper=True):
    rng = np.random.default_rng(seed)
    obs_t = np.linspace(DUR / (N_INV * PER), DUR, N_INV * PER)
    bi = dh.bin_index(obs_t)
    A3 = rng.uniform(0.2, 1.0, (len(obs_t), NP, N_INV))
    if upper:  # emission of bin j reaches only observations at or after bin j
        A3 *= (bi[:, None, None] >= np.arange(N_INV)[None, None, :])
    A = (A3 / REF[None, :, None]).reshape(len(obs_t), -1)
    theta = rng.uniform(0.5, 2.0, (NP, N_INV)) * REF[:, None]
    y = A @ theta.reshape(-1)
    return A, y, np.full(len(y), 1e-3), obs_t, theta


def test_bin_index():
    assert list(dh.bin_index(np.array([1.0, DUR / 6, DUR / 6 + 1, DUR]))) == [0, 0, 1, 5]


def test_ct_ignores_earlier_columns_and_is_biased():
    A, y, sig, t, theta = _problem()
    prior = np.zeros((NP, N_INV))
    ct = dh.estimate("CT", A, y, sig, t, 1e-9, 0.0, prior)
    A2 = A.reshape(len(y), NP, N_INV).copy()
    A2[:, :, 0] *= 7.0  # change bin-0 columns everywhere
    ct2 = dh.estimate("CT", A2.reshape(len(y), -1), y, sig, t, 1e-9, 0.0, prior)
    assert np.allclose(ct[:, 1:], ct2[:, 1:], rtol=0, atol=1e-12 * REF.max())
    m1 = dh.estimate("M1", A, y, sig, t, 1e-9, 0.0, prior)
    assert np.abs(m1 - theta).max() / REF.max() < 1e-3  # full history recovers the truth
    assert np.abs(ct[:, 0] - theta[:, 0]).max() / REF.max() < 1e-3  # bin 0 has no earlier emissions
    assert np.abs(ct[:, 3:] - theta[:, 3:]).max() / REF.max() > 0.05  # ignoring earlier emissions biases later bins


def test_causal_uses_no_future_data():
    A, y, sig, t, _ = _problem(1)
    prior = np.full((NP, N_INV), 0.5) * REF[:, None]
    base = dh.estimate("CAUSAL", A, y, sig, t, 1.0, 10.0, prior)
    for b in (0, 2, 4):
        y2 = y.copy()
        y2[dh.bin_index(t) > b] += 0.05
        pert = dh.estimate("CAUSAL", A, y2, sig, t, 1.0, 10.0, prior)
        assert np.abs(pert[:, b] - base[:, b]).max() <= 1e-12 * REF.max()
    assert np.abs(pert[:, 5] - base[:, 5]).max() > 0  # the last bin does use the late data


def test_retro_equals_fit_history_m2():
    A, y, sig, t, _ = _problem(2)
    prior = np.full((NP, N_INV), 0.5) * REF[:, None]
    r = dh.estimate("RETRO", A, y, sig, t, 1.0, 10.0, prior)
    ref = dyn.fit_history(A, y, sig, NP, N_INV, "M2", alpha=1.0, beta=10.0, theta0=prior, ref=REF).theta
    assert np.array_equal(r, ref)


def test_detect_exact_delay():
    s = np.array([0, 0, 0.4, 0.6, 1, 1.0])
    assert dh.detect(s, 0.0, 1.0, 2, 6) == (1.0, False)
    assert dh.detect(np.array([0, 0, 1, 1, 0, 0.0]), 0.0, 1.0, 2, 4) == (0.0, False)  # window ends before the pulse does
    assert dh.detect(np.array([0, 0, 1, 0, 1, 1.0]), 0.0, 1.0, 2, 6) == (2.0, False)  # must stay across
    d, miss = dh.detect(np.zeros(6), 0.0, 1.0, 2, 6)
    assert miss and np.isnan(d)
    assert dh.detect(np.array([1, 1, 1, 0.6, 0.2, 0.0]), 1.0, 0.0, 2, 6) == (2.0, False)  # downward step


def test_change_spec():
    h = np.full((NP, N_FINE), 1e5)
    h[3:] = 0.0
    h[0, :4] = 0.0
    assert dh.change_spec("activation", h) == (0, 4, N_FINE, 0.0, 1e5)
    h[3, 5:7] = 0.1
    assert dh.change_spec("pulse", h) == (3, 5, 7, 0.0, 0.1)
    assert dh.change_spec("constant", h) is None


def test_freeze_guard(tmp_path):
    frozen = {"L4|0.02": {"M0": [[1.0, 0.0]]}}
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"hash": _hash(frozen), "frozen": frozen}))
    assert dh.load_frozen(p)["L4|0.02"]["M0"] == [(1.0, 0.0)]
    h = _hash(frozen)
    frozen["L4|0.02"]["M0"] = [[2.0, 0.0]]  # modified after hashing
    p.write_text(json.dumps({"hash": h, "frozen": frozen}))
    with pytest.raises(RuntimeError, match="freeze guard"):
        dh.load_frozen(p)
    with pytest.raises(RuntimeError, match="freeze guard"):
        dh.test("", tmp_path, 1)


def test_test_episodes():
    eps = dh.episodes_test()
    assert len(eps) == 96 and eps[0]["seed"] == 7600 and eps[-1]["seed"] == 7695
