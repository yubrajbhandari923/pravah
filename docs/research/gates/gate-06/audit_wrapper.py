"""Gate-06: pre-E06 greedy_design (a4f4827, copied as old_identifiability.py) vs the new wrapper on random problems."""
import sys
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.cwd() / "src")); sys.path.insert(0, str(here))
import old_identifiability as old
from pravah.identifiability import greedy_design
same = 0
for s in range(50):
    r = np.random.default_rng(s); n = 36
    G3, xy = np.abs(r.normal(size=(n, 5, 3))) * r.uniform(0, 1, (n, 1, 1)) ** 3, r.uniform(0, 1e4, (n, 2))
    same += old.greedy_design(G3, xy, 6, 0.3, 2.0, 1500.0) == greedy_design(G3, xy, 6, 0.3, 2.0, 1500.0)
msg = f"old vs new greedy_design identical in {same}/50 random problems"; print(msg); (here / "audit_wrapper.out").write_text(msg + "\n")
