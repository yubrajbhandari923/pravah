"""Contract for pravah.placement (E06). Synthetic matrices only; the scenario adapter is checked with a stub."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg

from pravah import placement as P


def toy(n=10, H=3, K=3, seed=0, sep=0.0, sigma=0.5):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10000, (n, 2))
    return P.Problem(xy=xy, cand=np.arange(n), min_sep=sep, a_ref=2.0, sigma=sigma, G3=np.abs(rng.normal(size=(n, H, K))),
                     snapshots=rng.normal(size=(n, 40)) @ np.diag(np.linspace(3, .1, 40)), source_xy=rng.uniform(0, 10000, (K, 2)))


def lattice(m=6, spacing=625.0, sep=1250.0, seed=1):
    ii, jj = np.divmod(np.arange(m * m), m)
    xy = np.stack([ii, jj], 1) * spacing
    rng = np.random.default_rng(seed)
    return P.Problem(xy=xy, cand=np.arange(m * m)[::-1].copy(), min_sep=sep, a_ref=1.0, sigma=np.full(m * m, 0.3),
                     G3=np.abs(rng.normal(size=(m * m, 4, 2))), snapshots=rng.normal(size=(m * m, 30)),
                     source_xy=np.array([[1000.0, 1000.0], [2500.0, 2000.0]]))


@pytest.mark.parametrize("name", list(P.STRATEGIES))
@pytest.mark.parametrize("k", [1, 2, 4, 6])
def test_designs_valid_with_binding_separation(name, k):
    p = lattice()
    S = P.design(name, p, k, seed=3)
    assert len(S) == k == len(set(S)) and set(S) <= set(p.cand.tolist())
    assert all(np.hypot(*(p.xy[a] - p.xy[b])) >= p.min_sep for a, b in itertools.combinations(S, 2))
    assert S == P.design(name, p, k, seed=3)  # deterministic


def test_infeasible_budget_raises():
    p = lattice(m=3)  # 3x3 at 625 m, sep 1250: at most 4 sites
    for name in P.STRATEGIES:
        with pytest.raises(ValueError):
            P.design(name, p, 5)


@pytest.mark.parametrize("value", [P.dopt, P.aopt])
def test_greedy_nested_and_monotone(value):
    p = toy(n=12)
    vals, prev = [], []
    for k in range(1, 9):
        S = P.greedy(p, k, value)
        assert S[:-1] == prev
        prev = S
        vals.append(value(p, S))
    assert np.all(np.diff(vals) >= -1e-12)


def test_coherence_of_diversity_nondecreasing_in_k():
    p = toy(n=12)
    v = [P.coherence(p, P.diversity_design(p, k, tau=0.0)) for k in range(1, 8)]
    assert np.all(np.diff(v) >= -1e-12)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("k", [2, 3])
def test_greedy_vs_brute_force(seed, k):
    p = toy(n=10, seed=seed)
    g = P.dopt(p, P.greedy(p, k, P.dopt))
    best, S, nf = P.brute_force(p, k, P.dopt)
    assert nf == len(list(itertools.combinations(range(10), k))) and best >= g - 1e-12
    assert g >= (1 - 1 / np.e) * best  # Bayesian D is monotone submodular with f(empty)=0 and no binding constraint
    # QR's own objective: greedy pivots vs exhaustive log det
    q = P.podqr_value(p, P.podqr_design(p, k))
    assert P.brute_force(p, k, P.podqr_value)[0] >= q - 1e-12
    # diversity's own objective (smaller is better)
    d = P.coherence(p, P.diversity_design(p, k, tau=0.0))
    assert P.brute_force(p, k, P.coherence, maximize=False)[0] <= d + 1e-12


def test_brute_force_respects_separation():
    p = lattice(m=4)
    _, S, nf = P.brute_force(p, 3, P.dopt)
    allc = list(itertools.combinations(range(16), 3))
    assert nf < len(allc) and all(np.hypot(*(p.xy[a] - p.xy[b])) >= p.min_sep for a, b in itertools.combinations(S, 2))


def test_qr_matches_scipy_reference():
    rng = np.random.default_rng(5)
    Psi = np.linalg.qr(rng.normal(size=(30, 6)))[0]
    p = P.Problem(xy=np.zeros((30, 2)), cand=np.arange(30), min_sep=0.0)
    piv = scipy.linalg.qr(Psi.T, pivoting=True)[2]
    assert P.qr_select(Psi, p, 6) == piv[:6].tolist()


def test_qr_matches_pysensors():
    ps = pytest.importorskip("pysensors")
    p = toy(n=25, seed=7)
    k = 5
    ours = P.podqr_design(p, k)
    # optimizer level, on our basis
    assert ps.optimizers.QR().fit(P.pod_basis(p.snapshots, k)).get_sensors()[:k].tolist() == ours
    # end to end: SSPOR with an SVD basis on the same centred training snapshots (samples x features)
    X = (p.snapshots - p.snapshots.mean(1, keepdims=True)).T
    m = ps.SSPOR(basis=ps.basis.SVD(n_basis_modes=k, algorithm="arpack"), n_sensors=k).fit(X)
    assert m.get_selected_sensors().tolist() == ours  # pivots depend only on span(Psi), not its rotation/sign


def test_podqr_undefined_beyond_above_noise_modes():
    rng = np.random.default_rng(4)
    n = 16
    X = np.outer(rng.normal(size=n), rng.normal(size=50)) * 10 + rng.normal(size=(n, 50)) * 1e-6  # one strong mode + numerical dust
    p = P.Problem(xy=rng.uniform(0, 1e4, (n, 2)), cand=np.arange(n), min_sep=0.0, sigma=0.5, snapshots=X)
    assert p.r_eff == 1 and len(P.podqr_design(p, 1)) == 1
    with pytest.raises(ValueError, match="r_eff"):
        P.podqr_design(p, 2)
    assert np.isfinite(P.podqr_value(p, [0, 1, 2]))  # evaluation uses min(|S|, r_num) modes


def test_footprint_identical_pair_is_redundant():
    G3 = np.zeros((3, 2, 2))
    G3[0] = G3[1] = [[1.0, 0.5], [0.8, 0.4]]  # identical footprints (sites 0 and 1), far apart
    G3[2] = [[0.0, 0.3], [0.1, 0.2]]          # new direction, weaker
    p = P.Problem(xy=np.array([[0, 0], [5000, 0], [0, 5000.0]]), cand=np.arange(3), min_sep=1000.0, a_ref=1.0, sigma=0.1, G3=G3)
    rank = lambda S: np.linalg.matrix_rank(p.A3[S].reshape(-1, 2))
    assert rank([0, 1]) == rank([0]) == 1 and rank([0, 2]) == 2  # duplicate adds no new direction
    assert P.coherence(p, [0, 1]) == pytest.approx(1.0)
    S = P.greedy(p, 2, P.dopt)
    assert sorted(S) == [0, 2]  # D-opt does not choose the duplicate
    assert P.diversity_design(p, 2) == [0, 2]
    # under independent noise the duplicate still averages noise along the same direction: gain > 0 but < the new direction's
    assert 0 < P.dopt(p, [0, 1]) - P.dopt(p, [0]) < P.dopt(p, [0, 2]) - P.dopt(p, [0])


def test_criteria_values():
    G3 = np.zeros((2, 1, 2))
    G3[0, 0], G3[1, 0] = [3.0, 0.0], [0.0, 4.0]
    p = P.Problem(xy=np.array([[0, 0], [9e3, 0]]), cand=np.arange(2), min_sep=0.0, a_ref=1.0, sigma=1.0, G3=G3)
    assert P.dopt(p, [0, 1]) == pytest.approx(np.log(10 * 17))
    assert P.aopt(p, [0, 1]) == pytest.approx(1 - (1 / 10 + 1 / 17) / 2)
    assert P.eopt(p, [0, 1]) == pytest.approx(np.log(10))
    assert P.dopt(p, []) == 0 and P.aopt(p, []) == 0 and P.coherence(p, [0, 1]) == 0


def test_identifiability_wrapper_unchanged():
    from pravah.identifiability import greedy_design
    p = toy(n=9, sep=2000.0)
    assert greedy_design(p.G3, p.xy, 3, 0.5, 2.0, 2000.0) == P.greedy(p, 3, P.dopt)


def test_podqr_uses_training_hours_only():
    rng = np.random.default_rng(2)
    n, train, other = 20, np.arange(0, 48), np.arange(48, 96)
    st = rng.normal(size=(n, 96)) @ np.diag(np.linspace(2, .1, 96))

    def stub(st):
        return SimpleNamespace(truth=SimpleNamespace(site_truth=st, source_xy=np.zeros((1, 2))),
                               split_hours=lambda s: {"train": train, "test": other[24:], "validation": other[:24]}[s])

    poisoned = st.copy()
    poisoned[:, other] = np.nan  # any read outside the train split would propagate NaN and raise in pod_basis
    snap = P.training_snapshots(stub(poisoned))
    assert snap.shape == (n, 48) and np.isfinite(snap).all()
    changed = st.copy()
    changed[:, other] = rng.normal(size=(n, 48)) * 100
    pa = P.Problem(xy=rng.uniform(0, 1e4, (n, 2)), cand=np.arange(n), min_sep=0.0, sigma=0.01, snapshots=P.training_snapshots(stub(st)))
    pb = P.Problem(xy=pa.xy, cand=pa.cand, min_sep=0.0, sigma=0.01, snapshots=P.training_snapshots(stub(changed)))
    assert P.podqr_design(pa, 5) == P.podqr_design(pb, 5)
