"""FiPy forward model for the Phase-1 2-D depth-averaged passive tracer (docs/research/pde-and-discretization.md).

Solves  c_t + u.grad c = div(K grad c) - lam (c - c_bg) + source  on a regular Cartesian grid, backward Euler,
central diffusion, first-order upwind convection. FiPy is the only PDE engine. State is the explicit array
c[i, j] (cell centre ((i+.5)dx, (j+.5)dy)), so one step is x_{n+1} = A x_n + d (affine, linear in `source`).
scheme="vanleer" (FiPy VanLeerConvectionTerm, gate-00 B3) is TVD but EXPLICIT in time, so it needs per-axis
Courant <= 1. It is Lie-split by axis (x advection, then y advection + diffusion/removal/source), because FiPy's
unsplit term leaves a -u_x u_y dt cross-diffusion that is anti-diffusive along oblique winds. Its limiter makes the
step nonlinear in c: superposition/Jacobian code must not assume linearity under it
(docs/research/gates/b3-advection-scheme.md).
Units: c g m^-3, u m s^-1, K m^2 s^-1, lam s^-1, source g m^-3 s^-1, dt s, lengths m.
"""
import numpy as np
from fipy import (CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, PeriodicGrid2D, TransientTerm,
                  UpwindConvectionTerm, VanLeerConvectionTerm, FaceVariable)
from fipy.solvers.scipy import LinearLUSolver

BCS = ("study", "noflux", "periodic")
SCHEMES = {"upwind": UpwindConvectionTerm, "vanleer": VanLeerConvectionTerm}
VANLEER_MAX_COURANT = 1.0  # explicit-convection stability bound, per axis (the scheme is dimensionally split)


def _check(nx, ny, Lx, Ly, dt, n_steps, u, K, lam, c_bg, bc, scheme):
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {tuple(SCHEMES)}, got {scheme!r}")
    if bc not in BCS:
        raise ValueError(f"bc must be one of {BCS}, got {bc!r}")
    if nx < 1 or ny < 1 or Lx <= 0 or Ly <= 0 or dt <= 0 or n_steps < 0:
        raise ValueError("need nx, ny >= 1, Lx, Ly, dt > 0, n_steps >= 0")
    if K < 0 or lam < 0 or c_bg < 0:
        raise ValueError("K, lam, c_bg must be >= 0")
    if np.shape(u) != (2,):
        raise ValueError("u must be a uniform (ux, uy); non-uniform wind needs a protocol amendment (doc 2)")
    if scheme == "vanleer" and max(abs(u[0]) * nx / Lx, abs(u[1]) * ny / Ly) * dt > VANLEER_MAX_COURANT:
        raise ValueError(f"vanleer convection is explicit: need max(|ux|/dx, |uy|/dy) dt <= {VANLEER_MAX_COURANT}; reduce dt")


def simulate(*, nx, ny, Lx, Ly, dt, n_steps, u, K, lam, c_bg, c0, bc, scheme="upwind", source=None):
    """Advance c0 by n_steps of size dt; return c[nx, ny].

    source: volumetric term (1/h) sum theta_j b_j, g m^-3 s^-1, scalar or (nx, ny), must be >= 0, constant in time.
    """
    _check(nx, ny, Lx, Ly, dt, n_steps, u, K, lam, c_bg, bc, scheme)
    c0 = np.asarray(c0, float)
    if c0.shape != (nx, ny):
        raise ValueError(f"c0 must have shape {(nx, ny)}, got {c0.shape}")
    s = np.zeros((nx, ny)) if source is None else np.broadcast_to(np.asarray(source, float), (nx, ny))
    if (s < 0).any():
        raise ValueError("source must be nonnegative")
    dx, dy = Lx / nx, Ly / ny
    mesh = (PeriodicGrid2D if bc == "periodic" else Grid2D)(nx=nx, ny=ny, dx=dx, dy=dy)
    to_cells = lambda a: np.asarray(a).T.ravel()  # [i, j] -> FiPy index i + j*nx
    c = CellVariable(mesh=mesh, value=to_cells(c0), hasOld=True)
    S = CellVariable(mesh=mesh, value=to_cells(s))
    rates = [0.0, 0.0]  # implicit outflow loss rate through x walls, y walls
    coef =np.array([u[0], u[1]], float)[:, None] * np.ones(mesh.numberOfFaces)
    if bc == "noflux":
        coef = coef * ~np.asarray(mesh.exteriorFaces)  # closed box: no advective flux through walls
    elif bc == "study":
        # inflow faces (u.n < 0): Dirichlet c_bg, which also fixes the half-cell diffusive flux. Outflow and u.n = 0: free.
        n = np.asarray(mesh.faceNormals)
        inflow = np.asarray(mesh.exteriorFaces) & ((n * np.array(u, float)[:, None]).sum(0) < 0)
        c.constrain(c_bg, where=inflow)
        # FiPy drops the upwind flux on unconstrained exterior faces, so add the outflow (u.n >= 0) explicitly:
        # each such face removes (u.n) c_P A/V from its cell, an implicit loss rate (u.n)/cell-width.
        rx, ry = np.zeros((nx, ny)), np.zeros((nx, ny))
        rx[-1, :] += max(u[0], 0) / dx
        rx[0, :] += max(-u[0], 0) / dx
        ry[:, -1] += max(u[1], 0) / dy
        ry[:, 0] += max(-u[1], 0) / dy
        rates = [CellVariable(mesh=mesh, value=to_cells(rx)), CellVariable(mesh=mesh, value=to_cells(ry))]
    conv = lambda mask: SCHEMES[scheme](coeff=FaceVariable(mesh=mesh, rank=1, value=coef * np.array(mask, float)[:, None]))
    rhs = lambda rate: DiffusionTerm(coeff=K) - ImplicitSourceTerm(coeff=lam + rate) + lam * c_bg + S
    if scheme == "upwind":
        eqs = [TransientTerm() + conv((1, 1)) == rhs(rates[0] + rates[1])]
    else:  # Lie dimensional splitting: x advection, then y advection + the rest (removes the unsplit cross term, b3 note)
        eqs = [TransientTerm() + conv((1, 0)) == -ImplicitSourceTerm(coeff=rates[0]), TransientTerm() + conv((0, 1)) == rhs(rates[1])]
    solver = LinearLUSolver()  # direct solve, doc 5.4
    for _ in range(n_steps):
        for eq in eqs:
            c.updateOld()
            eq.solve(var=c, dt=dt, solver=solver)
    return c.value.reshape(ny, nx).T.copy()


def observe(c, cells):
    """Sampling operator H_S: exact cell-centre values at integer cells [(i, j), ...]; no interpolation."""
    c = np.asarray(c)
    idx = np.asarray(cells, int).reshape(-1, 2)
    if (idx < 0).any() or (idx[:, 0] >= c.shape[0]).any() or (idx[:, 1] >= c.shape[1]).any():
        raise IndexError("observation cell outside grid")
    return c[idx[:, 0], idx[:, 1]]
