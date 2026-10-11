# EXP-023 control: model_error_K1.2 (production)

Control (c): truth (validation and test) generated with K x 1.2 (declared one-parameter model error) while every estimator uses the nominal K; layouts are designed under the nominal model. Directions {-30,-10,0,10,30}; 10 random layouts. Baseline with the same seeds and no model error: model_error_baseline. EXP-023 control of F-029 requested by the independent review; synthetic; inverse crime except where the control breaks it. New seeds disjoint from the production, pilot, smoke and all other control configs.

Declared control (frozen in `freeze.json` before the test): `{"name": "model_error_K1.2", "truth_K_factor": 1.2}`.
Raw rows `raw/`; standard tables/figures per setting in `tables/<setting>/`, `figures/<setting>/` (flat when there is one setting);
`tables/control_gaps.csv` (A-B, C-B paired RMSE gaps), `tables/control_calibration.csv` (cov90, chi2/dof, chi2 reject rate).
