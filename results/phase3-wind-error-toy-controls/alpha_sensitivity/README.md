# EXP-023 control: alpha_sensitivity (production)

Control (a): strategies A, B, C each at alpha frozen on B's validation (EXP-019 rule), alpha chosen on each strategy's OWN validation RMSE, and alpha in {0.1, 1, 10}. Layouts (designed under B's alpha) and the prior are those of EXP-019; direction block only; 10 random layouts per budget instead of 20 (declared runtime reduction). EXP-023 control of F-029 requested by the independent review; synthetic; inverse crime except where the control breaks it. New seeds disjoint from the production, pilot, smoke and all other control configs.

Declared control (frozen in `freeze.json` before the test): `{"name": "alpha_sensitivity", "alpha_settings": ["frozen_B", "own", 0.1, 1.0, 10.0]}`.
Raw rows `raw/`; standard tables/figures per setting in `tables/<setting>/`, `figures/<setting>/` (flat when there is one setting);
`tables/control_gaps.csv` (A-B, C-B paired RMSE gaps), `tables/control_calibration.csv` (cov90, chi2/dof, chi2 reject rate).
