"""Bundled synthetic scenarios covering the required edge cases.

All measurements are synthetic and use the explicit additive model from the
task.  No biological or medical interpretation is attached to any value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

from .design import PublishedLayout, PlateDesign, Well
from .rational import fmt_rational
from .warehouse import FitReport, ReadoutWarehouse


@dataclass
class ScenarioResult:
    name: str
    description: str
    layout: PublishedLayout
    run_id: str
    measurement_count: int
    missing_used_wells: tuple[str, ...]
    report: FitReport
    truth: Mapping[str, object] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        complete = [s.sample_id for s in self.report.samples if s.status == "COMPLETE"]
        incomplete = [s.sample_id for s in self.report.samples if s.status == "INCOMPLETE"]
        return (
            f"rank {self.report.rank}/{self.report.n_params}, "
            f"nullity {self.report.nullity}, "
            f"missing {len(self.missing_used_wells)}, "
            f"complete {complete}, incomplete {incomplete}"
        )


def corrected_truth_statement(report: FitReport) -> dict:
    """Compact verifiable answer used by tests and the CLI."""

    return {
        "rank": report.rank,
        "n_parameters": report.n_params,
        "nullity": report.nullity,
        "parameter_unique": report.parameter_unique,
        "residual_sum_of_squares": fmt_rational(report.residual_sum_of_squares),
        "wells": {
            w.well: {
                "status": w.status,
                "corrected": None if w.corrected is None else fmt_rational(w.corrected),
                "bias_identifiable": w.bias_identifiable,
            }
            for w in report.wells
            if w.role == "sample"
        },
        "samples": {
            s.sample_id: {
                "status": s.status,
                "mean": None if s.mean is None else fmt_rational(s.mean),
                "range": None if s.range is None else fmt_rational(s.range),
                "reasons": [r["well"] + ":" + r["status"] for r in s.reasons],
            }
            for s in report.samples
        },
        "verification_ok": report.verification["ok"],
    }


def scenario_to_dict(result: ScenarioResult) -> dict:
    return {
        "name": result.name,
        "description": result.description,
        "headline": result.headline,
        "truth": {
            key: {
                k: fmt_rational(v) if isinstance(v, Fraction) else v
                for k, v in value.items()
            }
            if isinstance(value, Mapping)
            else fmt_rational(value)
            if isinstance(value, Fraction)
            else value
            for key, value in result.truth.items()
        },
        "layout": result.layout.to_dict(),
        "missing_used_wells": list(result.missing_used_wells),
        "report": result.report.to_dict(),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _full_control_design(
    plate_id: str, *, rows: int = 8, cols: int = 12, min_replicates: int = 2, max_replicates: int = 3
) -> PlateDesign:
    """Every well is a control; alternating blanks and rational references."""

    design = PlateDesign(
        plate_id, rows=rows, cols=cols, min_replicates=min_replicates, max_replicates=max_replicates
    )
    for row in range(rows):
        for col in range(cols):
            name = Well(row, col).name
            if (row + col) % 2 == 0:
                design.set_blank([name])
            else:
                design.set_reference([name], Fraction(row + 1, 2))
    return design


def _synthesise(
    layout: PublishedLayout,
    *,
    b: Fraction,
    row_effects: Mapping[int, Fraction],
    col_effects: Mapping[int, Fraction],
    sample_values: Mapping[str, Fraction],
    noise: Mapping[str, Fraction] | None = None,
) -> dict[str, Fraction]:
    noise = noise or {}
    measurements: dict[str, Fraction] = {}
    for well in layout.wells:
        if well.role == "unused":
            continue
        bias = b + row_effects[well.row] + col_effects[well.col]
        if well.role == "blank":
            base = Fraction(0)
        elif well.role == "reference":
            base = well.expected
        else:
            base = sample_values[well.sample_id]
        measurements[well.name] = base + bias + noise.get(well.name, Fraction(0))
    return measurements


# ---------------------------------------------------------------------------
# Scenario 1: complete controls, full identifiability
# ---------------------------------------------------------------------------


def scenario_complete_controls() -> ScenarioResult:
    design = _full_control_design("plate-complete")
    design.define_sample("S-alpha", ["A1", "B2"])
    design.define_sample("S-beta", ["C3", "D4"])
    design.define_sample("S-gamma", ["E5", "F6"])
    layout = design.publish()

    row_effects = {i: Fraction(2 * i - 7, 2) for i in range(8)}  # sums to 0
    col_effects = {j: Fraction(2 * j - 11, 4) for j in range(12)}  # sums to 0
    sample_values = {"S-alpha": Fraction(4), "S-beta": Fraction(9, 2), "S-gamma": Fraction(11, 3)}
    noise = {}
    # A single deterministic perturbation on one sample well keeps the fit
    # non-degenerate while leaving identifiability intact.
    noise["A1"] = Fraction(1, 5)
    measurements = _synthesise(
        layout,
        b=Fraction(2, 5),
        row_effects=row_effects,
        col_effects=col_effects,
        sample_values=sample_values,
        noise=noise,
    )
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="complete-controls")
    report = warehouse.fit(run)
    return ScenarioResult(
        name="complete_controls",
        description=(
            "Every well is a control, so all 19 bias parameters are identifiable. "
            "One sample well carries a deliberate +1/5 perturbation; its corrected "
            "value and the other, noise-free samples recover the generating values."
        ),
        layout=layout,
        run_id=run.run_id,
        measurement_count=len(measurements),
        missing_used_wells=run.missing_used_wells,
        report=report,
        truth={
            "b": Fraction(2, 5),
            "sample_values": sample_values,
            "row_effects": row_effects,
            "col_effects": col_effects,
        },
    )


# ---------------------------------------------------------------------------
# Scenario 2: disconnected row/column control graph
# ---------------------------------------------------------------------------


def scenario_disconnected_controls() -> ScenarioResult:
    design = PlateDesign("plate-disconnected", rows=8, cols=12, min_replicates=1, max_replicates=2)
    # Connected control component A covering rows A-B and columns 1-3.
    # Rows A and B are linked (col 1), columns 1-3 are linked through rows A/B.
    design.set_blank(["A1", "A2", "B1"])
    design.set_reference(["B3"], Fraction(6))
    # Disconnected component B: rows D-E x columns D-E.
    design.set_reference(["D4", "D5", "E4"], Fraction(6))
    design.set_blank(["E5"])
    # (A3) and (B2) are non-control cells whose row and column both belong to
    # component A, so their bias is identifiable even though the parameters
    # are not unique overall.
    design.define_sample("S-near", ["A3", "B2"])
    # A sample with no path to any control is not identifiable.
    design.define_sample("S-far", ["F7", "G8"])
    layout = design.publish()

    row_effects = {i: Fraction(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: Fraction(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-near": Fraction(3), "S-far": Fraction(8)}
    measurements = _synthesise(
        layout,
        b=Fraction(1, 3),
        row_effects=row_effects,
        col_effects=col_effects,
        sample_values=sample_values,
    )
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="disconnected-controls")
    report = warehouse.fit(run)
    return ScenarioResult(
        name="disconnected_controls",
        description=(
            "Two disconnected control components leave a large nullspace, so b/r/c "
            "are not unique.  Wells whose bias is invariant across all minimisers "
            "(S-near) are still corrected; S-far is UNIDENTIFIABLE with a witness "
            "direction that changes its bias by exactly one."
        ),
        layout=layout,
        run_id=run.run_id,
        measurement_count=len(measurements),
        missing_used_wells=run.missing_used_wells,
        report=report,
        truth={
            "b": Fraction(1, 3),
            "sample_values": sample_values,
            "row_effects": row_effects,
            "col_effects": col_effects,
        },
    )


# ---------------------------------------------------------------------------
# Scenario 3: parameters non-unique but a particular well bias is unique
# ---------------------------------------------------------------------------


def scenario_unique_bias_nonunique_params() -> ScenarioResult:
    design = _full_control_design(
        "plate-unique-bias", min_replicates=1, max_replicates=2
    )
    # Keep only three control cells in the top-left 2x2 block; everything
    # else becomes unused so the control graph is a 3-edge path.
    for row in range(8):
        for col in range(12):
            name = Well(row, col).name
            if name not in {"A1", "A2", "B1", "B2"}:
                design.clear_wells([name])
    design.set_blank(["A1", "A2"])
    design.set_reference(["B1"], Fraction(5))
    # (1,1) sits at the corner of the 2x2 square spanned by the controls.
    design.define_sample("S-corner", ["B2"])
    design.define_sample("S-far", ["D4", "E5"])
    layout = design.publish()

    b = Fraction(1, 2)
    # Existing rows are all 8, existing columns all 12; effects must be zero-sum.
    row_effects = {i: Fraction(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: Fraction(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-corner": Fraction(4), "S-far": Fraction(8)}
    measurements = _synthesise(
        layout,
        b=b,
        row_effects=row_effects,
        col_effects=col_effects,
        sample_values=sample_values,
    )
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="unique-bias-nonunique-params")
    report = warehouse.fit(run)
    corner_bias = b + row_effects[1] + col_effects[1]
    return ScenarioResult(
        name="unique_bias_nonunique_params",
        description=(
            "Only three control wells pin the field, so b/r/c are not unique.  The "
            "bias at the diagonal corner (B2) is nevertheless invariant across all "
            "optimal solutions, and only wells of that kind may be corrected."
        ),
        layout=layout,
        run_id=run.run_id,
        measurement_count=len(measurements),
        missing_used_wells=run.missing_used_wells,
        report=report,
        truth={
            "b": b,
            "sample_values": sample_values,
            "not_unique": ["b", "r", "c"],
            "unique_well_bias": {"B2": corner_bias},
        },
    )


# ---------------------------------------------------------------------------
# Scenario 4: missing replicate
# ---------------------------------------------------------------------------


def scenario_missing_replicate() -> ScenarioResult:
    design = _full_control_design("plate-missing", min_replicates=2, max_replicates=2)
    design.define_sample("S-partial", ["A1", "B1"])
    design.define_sample("S-complete", ["C2", "D2"])
    layout = design.publish()

    row_effects = {i: Fraction(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: Fraction(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-partial": Fraction(6), "S-complete": Fraction(5)}
    measurements = _synthesise(
        layout,
        b=Fraction(1, 4),
        row_effects=row_effects,
        col_effects=col_effects,
        sample_values=sample_values,
    )
    del measurements["B1"]  # one replicate never measured
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="missing-replicate")
    report = warehouse.fit(run)
    return ScenarioResult(
        name="missing_replicate",
        description=(
            "S-partial has one replicate marked MISSING, so its statistics are "
            "INCOMPLETE with the missing well named; S-complete is still summarised. "
            "No replicate is silently dropped."
        ),
        layout=layout,
        run_id=run.run_id,
        measurement_count=len(measurements),
        missing_used_wells=run.missing_used_wells,
        report=report,
        truth={
            "b": Fraction(1, 4),
            "sample_values": sample_values,
            "dropped_measurement": "B1",
        },
    )


# ---------------------------------------------------------------------------
# Scenario 5: role / reference correction on a later version
# ---------------------------------------------------------------------------


def scenario_role_correction() -> ScenarioResult:
    design = _full_control_design("plate-correction", min_replicates=2, max_replicates=2)
    design.define_sample("S-one", ["A1", "B1"])
    design.define_sample("S-two", ["C2", "D2"])
    layout_v1 = design.publish()

    row_effects = {i: Fraction(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: Fraction(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-one": Fraction(5), "S-two": Fraction(7)}

    # Run 1 against version 1: the reference expectations used to generate the
    # data are the version-1 values.
    measurements_v1 = _synthesise(
        layout_v1,
        b=Fraction(1, 2),
        row_effects=row_effects,
        col_effects=col_effects,
        sample_values=sample_values,
    )
    warehouse = ReadoutWarehouse()
    run_v1 = warehouse.import_run(layout_v1, measurements_v1, run_id="correction-v1")
    report_v1 = warehouse.fit(run_v1)

    # The reference expectation is corrected downward: B2 was mislabelled at
    # 1.5 but is really 0.5.  The raw readouts do not change; only the layout
    # does, and the fit is genuinely redone against the new expectation.
    design.correct_reference(["B2"], Fraction(1, 2))
    # A role change: turn blank A2 into a reference of 0 (structurally a
    # different control) and back one blank well to unused then reassign.
    design.set_reference(["A2"], Fraction(0))
    layout_v2 = design.publish()

    # Re-use the *same observations* where the layout still covers them.
    measurements_v2 = dict(measurements_v1)
    measurements_v2["A2"] = measurements_v1["A2"]  # now a reference expectation 0
    run_v2 = warehouse.import_run(layout_v2, measurements_v2, run_id="correction-v2")
    report_v2 = warehouse.fit(run_v2)

    # Run 1 must still be bound to version 1's layout and its original controls.
    assert run_v1.layout.digest == layout_v1.digest
    assert warehouse.fit(run_v1).layout_version == layout_v1.version
    assert run_v1.layout.digest != layout_v2.digest
    assert report_v1.well("B2").expected == layout_v1.well_map["B2"].expected
    assert report_v2.well("B2").expected == Fraction(1, 2)

    return ScenarioResult(
        name="role_correction",
        description=(
            "A corrected reference value plus a blank-to-reference role change "
            "produce a new layout version.  The same raw readouts are genuinely "
            "re-fitted against the new controls, while the earlier run stays bound "
            "to its old layout digest and expectations."
        ),
        layout=layout_v2,
        run_id=run_v2.run_id,
        measurement_count=len(measurements_v2),
        missing_used_wells=run_v2.missing_used_wells,
        report=report_v2,
        truth={
            "b": Fraction(1, 2),
            "sample_values": sample_values,
            "layout_v1_version": layout_v1.version,
            "layout_v1_digest": layout_v1.digest,
            "layout_v2_version": layout_v2.version,
            "layout_v2_digest": layout_v2.digest,
            "v1_B2_expected": layout_v1.well_map["B2"].expected,
            "v2_B2_expected": layout_v2.well_map["B2"].expected,
            "v1_rank": report_v1.rank,
            "v2_rank": report_v2.rank,
            "v1_rss": fmt_rational(report_v1.residual_sum_of_squares),
            "v2_rss": fmt_rational(report_v2.residual_sum_of_squares),
        },
    )


# ---------------------------------------------------------------------------
# Scenario 6: unavailable wells
# ---------------------------------------------------------------------------


def scenario_blocked_wells() -> ScenarioResult:
    design = _full_control_design("plate-blocked", min_replicates=2, max_replicates=2)
    design.clear_wells(["H11", "H12", "H10", "G10"])
    design.block_wells(["H11", "H12"])
    design.define_sample("S-block", ["A1", "B1"])
    layout = design.publish()

    row_effects = {i: Fraction(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: Fraction(2 * j - 11, 4) for j in range(12)}
    sample_values = {"S-block": Fraction(5)}
    measurements = _synthesise(
        layout,
        b=Fraction(1, 2),
        row_effects=row_effects,
        col_effects=col_effects,
        sample_values=sample_values,
    )
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, measurements, run_id="blocked-wells")
    report = warehouse.fit(run)
    return ScenarioResult(
        name="blocked_wells",
        description=(
            "Unavailable wells are published as unused/blocked and cannot accept "
            "samples or measurements; available-well queries exclude them."
        ),
        layout=layout,
        run_id=run.run_id,
        measurement_count=len(measurements),
        missing_used_wells=run.missing_used_wells,
        report=report,
        truth={
            "blocked": ["H11", "H12"],
            "available_for_new_sample": [
                w.name for w in design.available_wells("S-new")
            ][:12],
        },
    )


def build_all_scenarios() -> dict[str, ScenarioResult]:
    builders = (
        scenario_complete_controls,
        scenario_disconnected_controls,
        scenario_unique_bias_nonunique_params,
        scenario_missing_replicate,
        scenario_role_correction,
        scenario_blocked_wells,
    )
    return {result.name: result for result in (builder() for builder in builders)}
