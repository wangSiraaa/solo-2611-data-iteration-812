"""End-to-end checks over the bundled scenarios.

These tests treat the scenarios as the acceptance suite: generating values are
known, so every identifiable sample correction must reproduce them exactly.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from plate_effects import scenarios
from plate_effects.rational import fmt_rational
from plate_effects.warehouse import FitReport, clear_caches


F = Fraction


@pytest.fixture(scope="module")
def results():
    clear_caches()
    return scenarios.build_all_scenarios()


def _bias(truth, row, col):
    return truth["b"] + truth["row_effects"][row] + truth["col_effects"][col]


def test_all_scenarios_verify(results):
    for name, result in results.items():
        report = result.report
        assert report.verification["ok"], name
        assert report.verification["normal_equations_zero"], name
        assert report.verification["identifiability_claims_verified"], name
        assert report.verification["sample_summaries_verified"], name
        assert report.verification["numerical_rank_agrees"], name
        assert report.rank == report.verification["exact_rank"], name


def test_complete_controls_recover_generating_values(results):
    result = results["complete_controls"]
    report = result.report
    assert report.parameter_unique
    assert report.rank == report.n_params == 19
    truth = result.truth
    for sample_id, value in truth["sample_values"].items():
        summary = report.sample(sample_id)
        assert summary.status == "COMPLETE"
        if sample_id == "S-alpha":
            # A1 was perturbed by +1/5 before correction.
            assert summary.range > 0
        else:
            assert summary.mean == value
            assert summary.range == 0


def test_disconnected_controls_split_by_identifiability(results):
    result = results["disconnected_controls"]
    report = result.report
    assert report.nullity > 0
    assert not report.parameter_unique
    truth = result.truth
    near = report.sample("S-near")
    assert near.status == "COMPLETE"
    assert near.mean == truth["sample_values"]["S-near"]
    far = report.sample("S-far")
    assert far.status == "INCOMPLETE"
    assert {r["status"] for r in far.reasons} == {"UNIDENTIFIABLE"}
    for well in report.unidentifiable_wells:
        assert report.well(well).proof is not None


def test_unique_bias_scenario(results):
    result = results["unique_bias_nonunique_params"]
    report = result.report
    assert not report.parameter_unique
    truth = result.truth
    corner = report.well("B2")
    assert corner.bias_identifiable
    assert corner.corrected == truth["sample_values"]["S-corner"]
    assert corner.bias == truth["unique_well_bias"]["B2"]
    assert report.sample("S-corner").status == "COMPLETE"
    assert report.sample("S-far").status == "INCOMPLETE"


def test_missing_replicate_scenario(results):
    result = results["missing_replicate"]
    report = result.report
    truth = result.truth
    assert report.well(truth["dropped_measurement"]).status == "MISSING"
    partial = report.sample("S-partial")
    assert partial.status == "INCOMPLETE"
    assert partial.reasons[0]["well"] == truth["dropped_measurement"]
    complete = report.sample("S-complete")
    assert complete.status == "COMPLETE"
    assert complete.mean == truth["sample_values"]["S-complete"]


def test_role_correction_scenario(results):
    result = results["role_correction"]
    report = result.report
    truth = result.truth
    assert truth["layout_v1_digest"] != truth["layout_v2_digest"]
    assert truth["v1_B2_expected"] != truth["v2_B2_expected"]
    assert report.well("B2").expected == truth["v2_B2_expected"]
    assert report.well("A2").role == "reference"
    assert report.well("A2").expected == 0
    for sample_id, value in truth["sample_values"].items():
        summary = report.sample(sample_id)
        assert summary.status == "COMPLETE"
        assert summary.mean == value


def test_blocked_wells_scenario(results):
    result = results["blocked_wells"]
    report = result.report
    truth = result.truth
    layout = result.layout
    for name in truth["blocked"]:
        well = layout.well_map[name]
        assert well.blocked
        assert well.role == "unused"
    assert truth["available_for_new_sample"]
    for name in truth["available_for_new_sample"]:
        assert not layout.well_map[name].blocked
        assert layout.well_map[name].role == "unused"
    assert report.sample("S-block").status == "COMPLETE"


def test_every_layout_well_has_provenance(results):
    for name, result in results.items():
        layout = result.layout
        assert len(layout.wells) == layout.rows * layout.cols
        for well in layout.wells:
            assert well.provenance, (name, well.name)
            assert well.name == well.well_name if hasattr(well, "well_name") else True


def test_scenario_json_is_round_trippable(results):
    import json

    for name, result in results.items():
        payload = scenarios.scenario_to_dict(result)
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["name"] == name
        assert restored["report"]["linear_system"]["rank"] == result.report.rank
        # Every value is published both exactly and numerically.
        for well in restored["report"]["wells"]:
            for field in ("observed", "bias", "corrected", "residual", "expected"):
                value = well[field]
                if value is not None:
                    assert set(value) == {"exact", "float"}
