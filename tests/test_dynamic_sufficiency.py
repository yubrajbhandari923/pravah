import numpy as np

from pravah.studies.toy import dynamic_sufficiency as su


def _fake_ops(seed=0, sigma=0.3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((len(su.CANDS), 5, 30)) * 0.5
    Mq = rng.standard_normal((40, 30))
    return A, {"w": {"G": np.einsum("cti,ctj->cij", A, A) / sigma ** 2, "Q": Mq.T @ Mq}}, Mq


def test_closed_form_risk_matches_monte_carlo():
    sigma = 0.3
    A, ops, Mq = _fake_ops(sigma=sigma)
    x = np.abs(np.random.default_rng(1).standard_normal(30))
    C = {"w": np.outer(x, x)}
    j = su.risks_all(ops, 2, [(C, {"w": 1}, 1)])[0]
    lay = su.layouts_of(2)
    i = 123
    AS = A[lay[i]].reshape(-1, 30)
    H = np.eye(30) + AS.T @ AS / sigma ** 2
    rng = np.random.default_rng(2)
    eps = rng.standard_normal((40000, AS.shape[0])) * sigma
    est = ((AS @ x)[None] + eps) @ AS / sigma ** 2 @ np.linalg.inv(H)  # ridge, prior I, zero mean
    mc = np.mean(np.sum(((est - x) @ Mq.T) ** 2, axis=1))
    assert abs(mc - j[i]) / j[i] < 0.03


def test_risk_is_linear_in_the_second_moment_and_ensembles_independent():
    _, ops, _ = _fake_ops()
    rng = np.random.default_rng(3)
    X = np.abs(rng.standard_normal((6, 30)))
    Cs = [{"w": np.outer(x, x)} for x in X]
    each = [su.risks_all(ops, 2, [(c, {"w": 1}, 1)])[0] for c in Cs]
    joint = su.risks_all(ops, 2, [({"w": sum(c["w"] for c in Cs)}, {"w": 6}, 6)])[0]
    assert np.allclose(joint, np.mean(each, axis=0), rtol=1e-9)


def test_streams_are_deterministic_and_nested_free():
    import pravah.studies.toy.dynamic_study as ds
    lives = {w: np.ones((5, 12), bool) for w in ds.WINDS}
    a, ca = su.episode_moments(12, (5, 12, 0), lives)
    b, cb = su.episode_moments(12, (5, 12, 0), lives)
    c, _ = su.episode_moments(12, (5, 12, 1), lives)
    assert all(np.array_equal(a[w], b[w]) for w in a) and ca == cb
    assert not all(np.array_equal(a[w], c[w]) for w in a)
