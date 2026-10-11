# EXP-023 control: offset_prior (production)

Control (b): prior members at -17.5,-7.5,2.5,12.5,22.5 deg x speed {0.8,1,1.2}. Rule (declared before test, no test quantity used): shift the EXP-019 grid by a quarter of its 10 deg spacing (2.5 deg) and keep the speed factors; all truth directions of the grid and of the validation winds are multiples of 5 deg and all members are 2.5 mod 5, so no truth or validation wind is a member (asserted at run time; min distance 2.5 deg). The prior is asymmetric about 0 and the nominal wind (0) is not a member. Full EXP-019 grid. EXP-023 control of F-029 requested by the independent review; synthetic; inverse crime except where the control breaks it. New seeds disjoint from the production, pilot, smoke and all other control configs.

Declared control (frozen in `freeze.json` before the test): `{"name": "offset_prior", "off_prior": true}`.
Raw rows `raw/`; standard tables/figures per setting in `tables/<setting>/`, `figures/<setting>/` (flat when there is one setting);
`tables/control_gaps.csv` (A-B, C-B paired RMSE gaps), `tables/control_calibration.csv` (cov90, chi2/dof, chi2 reject rate).
