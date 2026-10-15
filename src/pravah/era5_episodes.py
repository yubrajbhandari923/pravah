"""ERA5 episode wind histories for the Belauri dynamic study (DYN-08): window, stratify, split by calendar day, select, clip.

ERA5 is ~28 km and spatially UNIFORM here (``pravah.era5`` caveats apply): a window is only a *large-scale direction/speed history*
(rotations, reversals, calms) at the Belauri centroid, never a local wind.  Everything is pure NumPy on an ``Era5Wind`` and testable
on the tracked-by-nobody extract ``data/era5/belauri_era5_10m_wind.nc`` (Jan + Jul 2023, hourly).

Rule (frozen in docs/dyn08_plan.md BEFORE any layout is designed):
* windows = non-overlapping ``hours``-long blocks starting at 00/08/16 UTC inside one calendar month (UTC, not Nepal time);
* split by the calendar DAY of the window start, never by window: days 1-10 design, 11-20 validation, 21-31 test (both months), so
  autocorrelated neighbouring windows cannot leak across splits;
* stratum = month x {steady: direction change < ``turn_deg`` | turning} x {speed < median | >= median} (8 strata; the median is
  the all-window median of the mean speed and is reported);
* a seeded round-robin over a seeded stratum order draws ``n`` windows per split, so every non-empty stratum is used before any repeats;
* speeds are clipped to ``vmax`` (direction kept) to respect the Courant guideline of the 500 m / dt 100 mesh; clipped hours are counted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .era5 import Era5Wind, from_deg, sample_at

SPLITS = {"design": (1, 10), "validation": (11, 20), "test": (21, 31)}


@dataclass(frozen=True, eq=False)
class Window:
    index: int           # position in the record of the first hour
    month: int
    day: int
    start: str           # ISO time of the first hour (UTC)
    u: np.ndarray        # (hours,) m/s toward +x, unclipped
    v: np.ndarray        # (hours,) m/s toward +y, unclipped
    mean_speed: float
    from_deg: float      # vector-mean 'from' bearing
    turn_deg: float      # angle between the vector means of the first and the last third of the window (0-180)

    @property
    def split(self) -> str:
        return split_of(self.day)


def split_of(day: int) -> str:
    for name, (a, b) in SPLITS.items():
        if a <= day <= b:
            return name
    raise ValueError(f"day {day} outside 1-31")


def _angle_between(u1: float, v1: float, u2: float, v2: float) -> float:
    n1, n2 = float(np.hypot(u1, v1)), float(np.hypot(u2, v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0  # a calm third has no direction: not counted as a turn
    c = (u1 * u2 + v1 * v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def make_windows(w: Era5Wind, lon: float, lat: float, hours: int = 8) -> list[Window]:
    """All non-overlapping ``hours``-blocks aligned to 00:00 UTC that stay inside one calendar month and are hourly-contiguous."""
    if 24 % hours:
        raise ValueError("hours must divide 24")
    u, v = sample_at(w, lon, lat)
    t = w.times.astype("datetime64[h]")
    out: list[Window] = []
    for i in range(len(t) - hours + 1):
        if int((t[i] - t[i].astype("datetime64[D]")) / np.timedelta64(1, "h")) % hours:
            continue
        blk = t[i:i + hours]
        if not (np.all(np.diff(blk) == np.timedelta64(1, "h")) and blk[0].astype("datetime64[M]") == blk[-1].astype("datetime64[M]")):
            continue
        uu, vv = u[i:i + hours], v[i:i + hours]
        m = max(hours // 3, 1)
        d0 = blk[0].astype("datetime64[D]")
        out.append(Window(i, int(blk[0].astype("datetime64[M]").astype(int) % 12 + 1), int((d0 - d0.astype("datetime64[M]")) / np.timedelta64(1, "D")) + 1,
                          str(blk[0]), uu, vv, float(np.hypot(uu, vv).mean()), float(from_deg(uu.mean(), vv.mean())),
                          _angle_between(uu[:m].mean(), vv[:m].mean(), uu[-m:].mean(), vv[-m:].mean())))
    return out


def speed_median(windows: list[Window]) -> float:
    return float(np.median([x.mean_speed for x in windows]))


def stratum(win: Window, median_speed: float, turn_deg: float = 45.0) -> str:
    return f"m{win.month:02d}/{'turning' if win.turn_deg >= turn_deg else 'steady'}/{'high' if win.mean_speed >= median_speed else 'low'}"


def strata_table(windows: list[Window], median_speed: float, turn_deg: float = 45.0) -> dict[str, dict[str, int]]:
    """{split: {stratum: n windows}} for the whole record (reported with every run)."""
    tab: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for x in windows:
        s = stratum(x, median_speed, turn_deg)
        tab[x.split][s] = tab[x.split].get(s, 0) + 1
    return tab


def select_windows(windows: list[Window], split: str, n: int, seed: int, median_speed: float, turn_deg: float = 45.0) -> list[Window]:
    """``n`` windows of ``split``: seeded round-robin over seeded-shuffled strata, each stratum shuffled once and consumed in order.
    Fewer than ``n`` are returned only if the split has fewer windows; strata that are empty are skipped (see ``strata_table``)."""
    pool = [x for x in windows if x.split == split]
    rng = np.random.default_rng([seed, 17])
    groups: dict[str, list[Window]] = {}
    for x in pool:
        groups.setdefault(stratum(x, median_speed, turn_deg), []).append(x)
    names = sorted(groups)
    rng.shuffle(names)
    for g in names:
        groups[g] = [groups[g][i] for i in rng.permutation(len(groups[g]))]
    out: list[Window] = []
    while len(out) < n and any(groups.values()):
        for g in names:
            if groups[g] and len(out) < n:
                out.append(groups[g].pop())
    return out


def clip_speed(u: np.ndarray, v: np.ndarray, vmax: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Scale (u, v) down to speed <= vmax where exceeded (direction kept); returns (u, v, number of clipped hours)."""
    sp = np.hypot(u, v)
    f = np.where(sp > vmax, vmax / np.maximum(sp, 1e-300), 1.0)
    return u * f, v * f, int((sp > vmax).sum())
