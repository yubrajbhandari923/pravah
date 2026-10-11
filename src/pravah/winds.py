"""PRESCRIBED toy winds for EXP-018 (idealised ridge deflection, delayed reversal, recirculation).

Every wind here is PRESCRIBED by hand.  It is not derived from terrain or from any meteorological model; a 2-D
incompressible stream function is not a realistic near-surface wind.  Builders are pure functions of a ``Domain`` and
return ``tuple[WindSegment, ...]`` (piecewise constant in time, optional cell-corner ``psi``; see docs/winds.md).
Coordinates: metres in the domain's own frame (origin at its south-west corner); ``u`` and ``v`` are the flow TOWARD +x / +y.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .config import Domain, WindSegment
from .geo.scenarios import eddy_streamfunction  # psi = u*y - v*x + A*exp(-r^2/2s^2); no Belauri data is read

LABEL = "prescribed wind, not terrain-derived"


def _corner_psi(domain: Domain, u: float, v: float, centre: tuple[float, float] | None, sigma: float,
                amplitude: float) -> np.ndarray:
    if centre is None or amplitude == 0.0:
        x = (np.arange(domain.nx + 1) * domain.dx)[None, :]
        y = (np.arange(domain.ny + 1) * domain.dy)[:, None]
        return u * y - v * x + 0.0 * (x + y)
    return eddy_streamfunction(domain, (0.0, 0.0), u, v, centre, sigma, amplitude)


def steady_uniform(domain: Domain, u: float, v: float = 0.0) -> tuple[WindSegment, ...]:
    return (WindSegment(0.0, u, v),)


def delayed_reversal(domain: Domain, u: float, v: float, t_r: float) -> tuple[WindSegment, ...]:
    """Uniform wind (u, v) until ``t_r`` seconds, then (-u, -v)."""
    if not t_r > 0:
        raise ValueError("t_r must be > 0")
    return (WindSegment(0.0, u, v), WindSegment(t_r, -u, -v))


def ridge_deflection(domain: Domain, u: float, v: float, centre: tuple[float, float], sigma: float, amplitude: float,
                     t_r: float | None = None) -> tuple[WindSegment, ...]:
    """psi = u*y - v*x + amplitude*exp(-r^2/2 sigma^2): the wind is bent around a Gaussian 'ridge' at ``centre``
    (peak deflection speed amplitude/(sigma*sqrt(e)) m/s); divergence free by construction.  With ``t_r`` the WHOLE field
    is reversed (psi -> -psi) at t_r (ridge + reversal)."""
    psi = _corner_psi(domain, u, v, centre, sigma, amplitude)
    segs = [WindSegment.from_streamfunction(0.0, psi, u, v)]
    if t_r is not None:
        segs.append(WindSegment.from_streamfunction(t_r, -psi, -u, -v))
    return tuple(segs)


def recirculation_eddy(domain: Domain, u: float, v: float, centre: tuple[float, float], sigma: float,
                       amplitude: float) -> tuple[WindSegment, ...]:
    """Same stream function with a strong bump: closed streamlines where the swirl speed amplitude/(sigma*sqrt(e))
    exceeds the background speed (the eddy of ``geo.scenarios.eddy_wind``)."""
    return ridge_deflection(domain, u, v, centre, sigma, amplitude)


# ---------------------------------------------------------------------------------------------- external data
def segments_from_series(domain: Domain, times: np.ndarray, *, psi: np.ndarray | None = None,
                         u: np.ndarray | None = None, v: np.ndarray | None = None) -> tuple[WindSegment, ...]:
    """Piecewise-constant (sample-and-hold) segments from an external series: ``times`` (nt,) s from 0, and either ``psi``
    (nt, ny+1, nx+1) at cell corners or cell-centre ``u``, ``v`` (nt, ny, nx) in m/s.  No time interpolation: value k holds
    from times[k] to times[k+1].  u, v are averaged to faces (edge faces copy the nearest cell) and the least-squares psi
    with d psi/dy = u_f, -d psi/dx = v_f is solved (lsqr), i.e. the divergence-free part is kept; the dropped divergent
    part is NOT zero for real winds and is discarded silently.  psi is defined up to a constant (set to 0 at [0, 0])."""
    t = np.asarray(times, dtype=float)
    ny, nx, dx, dy = domain.ny, domain.nx, domain.dx, domain.dy
    if psi is None:
        if u is None or v is None:
            raise ValueError("give psi, or both u and v")
        k = (ny + 1) * (nx + 1)
        idx = np.arange(k).reshape(ny + 1, nx + 1)
        r_u, r_v = idx[:-1, :].ravel(), idx[:, :-1].ravel()  # x-normal faces (ny, nx+1): (psi[j+1,i]-psi[j,i])/dy
        nu = ny * (nx + 1)
        rows = np.concatenate([np.arange(nu), np.arange(nu), nu + np.arange(len(r_v)), nu + np.arange(len(r_v))])
        cols = np.concatenate([idx[1:, :].ravel(), r_u, idx[:, 1:].ravel(), r_v])
        vals = np.concatenate([np.full(nu, 1 / dy), np.full(nu, -1 / dy),
                               np.full(len(r_v), -1 / dx), np.full(len(r_v), 1 / dx)])
        a = coo_matrix((vals, (rows, cols)), shape=(nu + len(r_v), k)).tocsr()
        ps = []
        for uk, vk in zip(np.asarray(u, float), np.asarray(v, float)):
            up = np.pad(uk, ((0, 0), (1, 1)), mode="edge")
            vp = np.pad(vk, ((1, 1), (0, 0)), mode="edge")
            b = np.concatenate([(0.5 * (up[:, :-1] + up[:, 1:])).ravel(), (0.5 * (vp[:-1] + vp[1:])).ravel()])
            p = lsqr(a, b, atol=1e-14, btol=1e-14, iter_lim=20000)[0].reshape(ny + 1, nx + 1)
            ps.append(p - p[0, 0])
        psi = np.array(ps)
    psi = np.asarray(psi, dtype=float)
    if psi.shape != (len(t), ny + 1, nx + 1):
        raise ValueError(f"psi must have shape {(len(t), ny + 1, nx + 1)}, got {psi.shape}")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must start at 0 and increase")
    # nominal (u, v) = domain-mean wind: mean of d psi/dy and -d psi/dx
    return tuple(WindSegment.from_streamfunction(float(tk), p, float(((p[1:] - p[:-1]) / dy).mean()),
                                                 float((-(p[:, 1:] - p[:, :-1]) / dx).mean()))
                 for tk, p in zip(t, psi))


def face_divergence(domain: Domain, psi: np.ndarray) -> np.ndarray:
    """Per-cell discrete divergence [1/s] of the face fluxes defined by psi (identically ~0 by construction)."""
    uf = (psi[1:, :] - psi[:-1, :]) / domain.dy
    vf = -(psi[:, 1:] - psi[:, :-1]) / domain.dx
    return (uf[:, 1:] - uf[:, :-1]) / domain.dx + (vf[1:, :] - vf[:-1, :]) / domain.dy


# ---------------------------------------------------------------------------------------------- EXP-021 channel
def deflected_channel(domain: Domain, u: float, x0: float, width: float, offset: float,
                      kind: str = "step") -> tuple[WindSegment, ...]:
    """psi = u*(y - h(x)): streamlines are y = const + h(x), so the flow is turned around an obstacle-like feature.
    u_wind = dpsi/dy = u and v_wind = -dpsi/dx = u*h'(x) (divergence free).  PRESCRIBED, not terrain physics: nothing is
    computed from a DEM and there is no pressure or blocking dynamics.  h(x) is a smooth step ("step": h = offset*(1+tanh((x-x0)/width))/2,
    net downstream displacement ``offset``, peak turning angle atan(offset/(2 width))) or a Gaussian bump ("bump": h = offset*exp(-(x-x0)^2/2 width^2),
    zero net displacement, peak angle atan(offset/(width sqrt(e)))).  Far upstream and downstream the wind is the uniform (u, 0)."""
    x = (np.arange(domain.nx + 1) * domain.dx)[None, :]
    y = (np.arange(domain.ny + 1) * domain.dy)[:, None]
    if kind == "step":
        h = 0.5 * offset * (1.0 + np.tanh((x - x0) / width))
    elif kind == "bump":
        h = offset * np.exp(-((x - x0) ** 2) / (2.0 * width**2))
    else:
        raise ValueError("kind must be 'step' or 'bump'")
    return (WindSegment.from_streamfunction(0.0, u * (y - h), u, 0.0),)
