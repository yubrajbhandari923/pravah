"""Small matplotlib helpers. Every function returns a Figure (or animation) and never calls show()."""
from __future__ import annotations

from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe default; a notebook backend already chosen is kept
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .config import Domain, GaussianSource  # noqa: E402
from .observations import SensorLayout  # noqa: E402

_LABEL = "concentration (microgram/m$^3$)"


def _fig(ncols: int = 1, nrows: int = 1, **kw: Any) -> tuple[Figure, list[Any]]:
    fig, axes = plt.subplots(nrows, ncols, **kw)
    return fig, list(np.atleast_1d(axes).ravel())


def _extent(conc: xr.DataArray) -> tuple[float, float, float, float]:
    x, y = conc["x"].values, conc["y"].values
    hx, hy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    return float(x[0] - hx), float(x[-1] + hx), float(y[0] - hy), float(y[-1] + hy)


def plot_sources(sources: Sequence[GaussianSource], domain: Domain, layout: SensorLayout | None = None) -> Figure:
    fig, (ax,) = _fig(figsize=(5, 5))
    for s in sources:
        ax.add_patch(Circle((s.x, s.y), 2 * s.sigma, fill=False))
        ax.plot(s.x, s.y, "r*")
        ax.annotate(s.name, (s.x, s.y), textcoords="offset points", xytext=(4, 4))
    if layout is not None:
        ax.plot(layout.x, layout.y, "k^", label="sensors")
        ax.legend()
    ax.set(xlim=(0, domain.lx), ylim=(0, domain.ly), aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title="Sources (circle = 2 sigma)")
    return fig


def plot_field_snapshots(conc: xr.DataArray, times: Sequence[float] | None = None,
                         layout: SensorLayout | None = None) -> Figure:
    ts = list(times) if times is not None else [float(conc["time"].values[i]) for i in
                                                 np.unique(np.linspace(0, conc.sizes["time"] - 1, 4).astype(int))]
    fig, axes = _fig(len(ts), figsize=(4 * len(ts), 4), squeeze=False, constrained_layout=True)
    vmax = float(conc.max())
    ext = _extent(conc)
    for ax, t in zip(axes, ts):
        ax.imshow(conc.sel(time=t, method="nearest").values, origin="lower", extent=ext, vmin=0, vmax=vmax)
        if layout is not None:
            ax.plot(layout.x, layout.y, "w^", mec="k")
        ax.set(title=f"t = {t:.0f} s", xlabel="x (m)", ylabel="y (m)")
    fig.colorbar(axes[0].images[0], ax=axes, label=_LABEL)
    return fig


def plot_traces(values: xr.DataArray, truth: xr.DataArray | None = None) -> Figure:
    """Time series per sensor (dots = observations, lines = optional noiseless truth), one panel per sensor."""
    ids = [str(s) for s in values["sensor"].values]
    fig, axes = _fig(len(ids), figsize=(4 * len(ids), 3), squeeze=False, sharey=True, constrained_layout=True)
    for ax, s in zip(axes, ids):
        ax.plot(values["time"], values.sel(sensor=s), "o", ms=3, label="observed")
        if truth is not None:
            ax.plot(truth["time"], truth.sel(sensor=s), "-", label="truth")
        ax.set(title=s, xlabel="t (s)")
    axes[0].set_ylabel(_LABEL)
    axes[0].legend()
    return fig


def plot_mass_budget(budget: xr.Dataset) -> Figure:
    fig, (ax, ax2) = _fig(2, figsize=(10, 4), constrained_layout=True)
    for name in ("emission", "removal", "inflow", "outflow", "mass_change"):
        ax.plot(budget["time"], budget[name], label=name)
    ax.set(xlabel="t (s)", ylabel="microgram per interval", title="Mass budget terms")
    ax.legend()
    ax2.plot(budget["time"], budget["residual"], "k-")
    ax2.set(xlabel="t (s)", ylabel="microgram", title="Budget residual")
    return fig


def plot_layout(candidates: SensorLayout, selected: SensorLayout, sources: Sequence[GaussianSource] = (),
                domain: Domain | None = None) -> Figure:
    fig, (ax,) = _fig(figsize=(5, 5))
    ax.plot(candidates.x, candidates.y, ".", c="0.6", label="candidates")
    ax.plot(selected.x, selected.y, "ro", label="selected")
    for i, s in enumerate(selected.ids):
        ax.annotate(s, (selected.x[i], selected.y[i]), textcoords="offset points", xytext=(4, 4))
    if sources:
        ax.plot([s.x for s in sources], [s.y for s in sources], "b*", ms=10, label="sources")
    if domain is not None:
        ax.set(xlim=(0, domain.lx), ylim=(0, domain.ly))
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Sensor layout")
    ax.legend()
    return fig


def plot_compare_fields(truth: xr.DataArray, est: xr.DataArray, time: float | None = None) -> Figure:
    """Truth, estimate and difference (est - truth) at one time (default: last)."""
    t = float(truth["time"].values[-1]) if time is None else time
    a = truth.sel(time=t, method="nearest").values
    b = est.sel(time=t, method="nearest").values
    fig, axes = _fig(3, figsize=(13, 4), constrained_layout=True)
    vmax = float(max(a.max(), b.max()))
    ext = _extent(truth)
    for ax, arr, title in zip(axes[:2], (a, b), ("truth", "estimate")):
        ax.imshow(arr, origin="lower", extent=ext, vmin=0, vmax=vmax)
        ax.set_title(f"{title}, t = {t:.0f} s")
    fig.colorbar(axes[0].images[0], ax=axes[:2], label=_LABEL)
    d = b - a
    m = float(np.abs(d).max()) or 1.0
    im = axes[2].imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-m, vmax=m)
    axes[2].set_title("estimate - truth")
    fig.colorbar(im, ax=axes[2])
    return fig


def animate_field(conc: xr.DataArray, interval_ms: int = 100) -> FuncAnimation:
    fig, (ax,) = _fig(figsize=(5, 4.5))
    im = ax.imshow(conc.isel(time=0).values, origin="lower", extent=_extent(conc), vmin=0, vmax=float(conc.max()) or 1.0)
    fig.colorbar(im, label=_LABEL)
    times = conc["time"].values

    def update(k: int) -> tuple[Any, ...]:
        im.set_data(conc.isel(time=k).values)
        ax.set_title(f"t = {times[k]:.0f} s")
        return (im,)

    return FuncAnimation(fig, update, frames=conc.sizes["time"], interval=interval_ms, blit=False)


def plot_error_vs_budget(df: pd.DataFrame, budget_col: str = "n_sensors", error_col: str = "error",
                         group_col: str = "method") -> Figure:
    """Mean error vs sensor budget per method; error bars are +/- 1 sample standard deviation across replicates
    (a spread, not a confidence interval)."""
    fig, (ax,) = _fig(figsize=(6, 4))
    for name, g in df.groupby(group_col):
        s = g.groupby(budget_col)[error_col].agg(["mean", "std"]).fillna(0.0)
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=str(name))
    ax.set(xlabel=budget_col, ylabel=f"mean {error_col}", title="Mean error, bars = +/- 1 s.d. across replicates")
    ax.legend()
    return fig
