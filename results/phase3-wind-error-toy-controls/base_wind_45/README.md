# EXP-023 control: base_wind_45 (production)

Control (d): base (nominal) wind rotated by 45 deg counter-clockwise; wind-error keys are relative to it. Directions {-30,-10,0,10,30}; 10 random layouts. EXP-023 control of F-029 requested by the independent review; synthetic; inverse crime except where the control breaks it. New seeds disjoint from the production, pilot, smoke and all other control configs.

Declared control (frozen in `freeze.json` before the test): `{"name": "base_wind_45", "base_rotation_deg": 45}`.
Raw rows `raw/`; standard tables/figures per setting in `tables/<setting>/`, `figures/<setting>/` (flat when there is one setting);
`tables/control_gaps.csv` (A-B, C-B paired RMSE gaps), `tables/control_calibration.csv` (cov90, chi2/dof, chi2 reject rate).
