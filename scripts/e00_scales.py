# Rescued 2026-11-12 from /tmp/t004 (gate-00 scale estimate behind blocker B1, legacy RUN-0001).
"""Back-of-envelope scales implied by preregistered-protocol.md defaults (no PDE solved)."""
import json, math, os
L=10000.0; n=64; dx=L/n; K=50.0; lam=1/(12*3600); H=500.0; dt=300.0
sig=400.0; aref=100.0; amin=0.5*aref; amax=5*aref; cbg=5.0
peak_flux=lambda a: a/(2*math.pi*sig**2)           # ug m^-2 s^-1
res={}
res["dx_m"]=dx
res["peak_source_conc_rate_aref_ug_m3_s"]=peak_flux(aref)/H
res["local_steady_scale_aref_ug_m3"]=peak_flux(aref)/H/lam   # no advection: upper bound on in-source anomaly
res["local_steady_scale_amax_ug_m3"]=peak_flux(amax)/H/lam
# plume centreline, uniform wind u, crosswind width W~2*sigma_y(x=5km), steady: c ~ a/(H*u*W)
for u in (0.5,3.0,5.0):
    W=2*math.sqrt(sig**2+2*K*5000/u)
    res[f"plume_anomaly_aref_u{u}_ug_m3"]=aref/(H*u*W)
    res[f"plume_anomaly_amax_u{u}_ug_m3"]=amax/(H*u*W)
res["noise_sigma_if_5pct_of_background_ug_m3"]=0.05*cbg
res["cell_peclet_u3"]=3*dx/K; res["cell_peclet_u5"]=5*dx/K
res["upwind_numerical_diffusion_u3_m2_s"]=3*dx/2; res["upwind_numerical_diffusion_u5_m2_s"]=5*dx/2
res["courant_u5_dt300"]=5*dt/dx
res["steps_per_sim_1488h"]=1488*3600/dt
res["test_issue_times"]=18*24-6
res["n_ge8_cells"]=3*3*4; res["cells_needed_70pct"]=math.ceil(0.7*36)
d=os.environ.get("RESEARCH_RUN_DIR")
for k,v in res.items():
    print(k,v)
    if d:
        with open(d+"/metrics.jsonl","a") as f: f.write(json.dumps({"name":k,"value":v,"step":0})+"\n")
