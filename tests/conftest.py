import matplotlib

matplotlib.use("Agg")  # headless tests; library code never sets the backend


# Tests measured >= 25 s (call or fixture setup) in the 2026-10-13 full run (1665 s total; these ~17 tests are ~1100 s of it).
# They are deselected by default (pyproject addopts "-m 'not slow'"); run everything with `pytest -m ""` before a release or after
# touching the forward solver, the inverse, the design objective or the regression-gate scripts.
SLOW = {
    "test_transfer.py::test_pipeline_runs_on_unlike_region_same_outputs_as_belauri",
    "test_convergence_stage2.py::test_workers_identical_to_sequential",
    "test_windunc.py::test_default_run_unchanged",
    "test_windunc.py::test_k_mismatch_truth_path",
    "test_geodesign_robust.py::test_refinement_stability_smoke",
    "test_geodesign_robust.py::test_layouts_reused_verbatim",
    "test_geodesign_robust.py::test_cli_end_to_end",
    "test_geodesign_robust.py::test_determinism",
    "test_transboundary.py::test_full_smoke_run_all_studies",
    "test_transboundary.py::test_recovery_outputs_and_noise_scale",
    "test_geodesign.py::test_smoke_cli_outputs_valid",
    "test_geodesign.py::test_sigma_is_fixed_and_independent_of_the_test_ensemble",
    "test_geodesign.py::test_constraints_respected_in_every_layout",
    "test_scheme_vanleer.py::test_gaussian_error_and_order_beat_legacy",
    "test_belauri.py::test_smoke_run_end_to_end_writes_all_files",
}


def pytest_collection_modifyitems(items):
    import pytest
    for item in items:
        if item.nodeid.split("tests/")[-1] in SLOW:
            item.add_marker(pytest.mark.slow)
