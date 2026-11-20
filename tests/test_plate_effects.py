"""Test suite for the plate design service and readout warehouse."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from plate_effects.design import (
    DesignError,
    LayoutValidationError,
    PlateDesign,
    PublishedLayout,
    PublishedLayoutError,
    Role,
    Well,
    WellError,
)
from plate_effects.warehouse import (
    ImportError_,
    ReadoutWarehouse,
    WarehouseError,
    _factor_for,
    _structure_for,
    clear_caches,
)


F = Fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def all_wells(rows: int = 8, cols: int = 12) -> list[str]:
    return [Well(r, c).name for r in range(rows) for c in range(cols)]


def make_full_control_design(
    plate_id: str = "P-full", *, min_replicates: int = 1, max_replicates: int | None = None
) -> PlateDesign:
    design = PlateDesign(
        plate_id,
        rows=8,
        cols=12,
        min_replicates=min_replicates,
        max_replicates=max_replicates,
    )
    for row in range(8):
        for col in range(12):
            name = Well(row, col).name
            if (row + col) % 2 == 0:
                design.set_blank([name])
            else:
                design.set_reference([name], F(row + 1, 2))
    return design


def synth(layout: PublishedLayout, b, row_effects, col_effects, sample_values):
    out = {}
    for well in layout.wells:
        if well.role == "unused":
            continue
        bias = b + row_effects[well.row] + col_effects[well.col]
        if well.role == "blank":
            base = F(0)
        elif well.role == "reference":
            base = well.expected
        else:
            base = sample_values[well.sample_id]
        out[well.name] = base + bias
    return out


def full_control_layout(
    plate_id: str = "P-full", *, min_replicates: int = 2, max_replicates: int = 3
) -> PublishedLayout:
    design = PlateDesign(
        plate_id, rows=8, cols=12, min_replicates=min_replicates, max_replicates=max_replicates
    )
    for row in range(8):
        for col in range(12):
            name = Well(row, col).name
            if (row + col) % 2 == 0:
                design.set_blank([name])
            else:
                design.set_reference([name], F(row + 1, 2))
    return design.publish()


# ---------------------------------------------------------------------------
# Rational kernel
# ---------------------------------------------------------------------------


def test_rref_and_rank_exact():
    from plate_effects.rational import nullspace, rank, solve_linear

    # Symbolic-looking matrix that a float solver would struggle with.
    eps = F(1, 10**12)
    matrix = [[F(1), F(1)], [F(1), F(1) + eps]]
    assert rank(matrix) == 2
    assert nullspace(matrix) == []


def test_rank_detects_exact_dependency():
    from plate_effects.rational import rank

    matrix = [[F(1), F(2), F(3)], [F(2), F(4), F(6)]]
    assert rank(matrix) == 1


def test_solve_linear_consistent_and_inconsistent():
    from plate_effects.rational import solve_linear

    matrix = [[F(1), F(1)], [F(1), F(-1)]]
    particular, nul, r, ok = solve_linear(matrix, [F(2), F(0)])
    assert ok and r == 2 and nul == []
    assert particular == [F(1), F(1)]

    bad, nul, r, ok = solve_linear([[F(1), F(1)], [F(1), F(1)]], [F(1), F(2)])
    assert not ok


def test_parse_rational_forms():
    from plate_effects.rational import RationalError, parse_rational

    assert parse_rational("3/4") == F(3, 4)
    assert parse_rational(0.1) == F(1, 10)
    assert parse_rational("-2.5") == F(-5, 2)
    with pytest.raises(RationalError):
        parse_rational("nan")
    with pytest.raises(RationalError):
        parse_rational(True)


# ---------------------------------------------------------------------------
# Design service
# ---------------------------------------------------------------------------


def test_design_bounds_and_geometry():
    with pytest.raises(DesignError):
        PlateDesign("P", rows=9, cols=12)
    with pytest.raises(DesignError):
        PlateDesign("P", rows=8, cols=13)
    with pytest.raises(DesignError):
        PlateDesign("P", rows=8, cols=12, min_replicates=3, max_replicates=2)


def test_well_parse_and_bounds():
    assert Well.parse("A1", 8, 12) == Well(0, 0)
    assert Well.parse("h12", 8, 12) == Well(7, 11)
    with pytest.raises(WellError):
        Well.parse("I1", 8, 12)
    with pytest.raises(WellError):
        Well.parse("A13", 8, 12)
    with pytest.raises(WellError):
        Well.parse("Z9", 8, 12)


def test_blank_is_an_observation_not_absence():
    design = PlateDesign("P", rows=2, cols=2)
    design.set_blank(["A1"])
    layout = design.publish()
    well = layout.well_map["A1"]
    assert well.role == "blank"
    assert well.expected == F(0)

    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout, {"A1": F(1, 2)})
    report = warehouse.fit(run)
    result = report.well("A1")
    assert result.observed == F(1, 2)
    assert result.status == "CONTROL"


def test_duplicate_replicates_within_sample_rejected():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    with pytest.raises(DesignError):
        design.define_sample("S", ["A1", "A1"])


def test_replicate_row_rule():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=3)
    with pytest.raises(LayoutValidationError):
        design.define_sample("S", ["A1", "A2"])  # shares row 0
    # Different rows are fine.
    design.define_sample("S", ["A1", "B2"])
    report = design.validate()
    assert report.ok
    assert report.sample("S").row_unique


def test_replicate_count_bounds_per_sample():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=3)
    design.set_sample_bounds("S", 2, 2)
    with pytest.raises(LayoutValidationError):
        design.define_sample("S", ["A1"])
    design.define_sample("S", ["A1", "B1"])
    assert design.validate().sample("S").replicate_count == 2
    with pytest.raises(LayoutValidationError):
        design.define_sample("S", ["A1", "B1", "C1"])


def test_available_wells_excludes_used_blocked_and_replicate_rows():
    design = PlateDesign("P", rows=4, cols=4, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "B1"])
    design.block_wells(["A2", "B2"])
    design.define_sample("S", ["C1"])
    available = {w.name for w in design.available_wells("S")}
    assert "A1" not in available  # used
    assert "A2" not in available  # blocked
    assert "B1" not in available  # used
    assert "B2" not in available  # blocked
    assert "C1" not in available  # used
    assert "C2" not in available  # same row as a replicate
    assert "D1" in available
    assert "D3" in available


def test_available_wells_empty_at_max_replicates():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.define_sample("S", ["A1", "B1"])
    assert design.available_wells("S") == ()


def test_failed_replace_layout_rolls_back_entirely():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "A2"])
    design.define_sample("S", ["B1", "C1"])
    before = design.publish()
    version_before = design.version
    roles_before = design.roles
    blocked_before = design.blocked

    with pytest.raises(LayoutValidationError):
        design.replace_layout(
            {
                "A1": Role.blank(),
                "B1": Role.sample("S"),
                "B2": Role.sample("S"),  # same row -> invalid
            }
        )
    assert design.version == version_before
    after = design.publish()
    assert after.digest == before.digest
    assert design.roles == roles_before
    assert design.blocked == blocked_before


def test_failed_replace_blocks_wells_rolls_back():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    before = design.publish()
    with pytest.raises(LayoutValidationError):
        design.replace_layout(
            {"A1": Role.blank()}, blocked=["A1"]  # control well cannot be blocked
        )
    assert design.publish().digest == before.digest


def test_role_change_is_a_real_refit():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "A2"])
    design.set_reference(["B1", "B2"], F(10))
    design.define_sample("S", ["C1", "D1"])
    layout_v1 = design.publish()
    assert layout_v1.well_map["A1"].role == "blank"

    design.set_reference(["A1"], F(4))
    layout_v2 = design.publish()
    assert layout_v2.digest != layout_v1.digest
    assert layout_v2.well_map["A1"].role == "reference"
    assert layout_v2.well_map["A1"].expected == F(4)

    row_effects = {i: F(0) for i in range(8)}
    col_effects = {j: F(0) for j in range(12)}
    measurements = synth(
        layout_v2,
        F(1),
        row_effects,
        col_effects,
        {"S": F(2)},
    )
    warehouse = ReadoutWarehouse()
    run = warehouse.import_run(layout_v2, measurements)
    report = warehouse.fit(run)
    assert report.well("A1").expected == F(4)
    assert report.well("A1").status == "CONTROL"


def test_old_run_remains_bound_to_old_layout():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "A2"])
    design.set_reference(["B1", "B2"], F(10))
    design.define_sample("S", ["C1", "D1"])
    v1 = design.publish()
    warehouse = ReadoutWarehouse()
    r1 = warehouse.import_run(v1, synth(v1, F(0), {i: F(0) for i in range(8)}, {j: F(0) for j in range(12)}, {"S": F(1)}))
    design.correct_reference(["B1", "B2"], F(12))
    v2 = design.publish()
    warehouse.import_run(v2, synth(v2, F(0), {i: F(0) for i in range(8)}, {j: F(0) for j in range(12)}, {"S": F(1)}))
    assert warehouse.fit(r1).layout_digest == v1.digest
    assert warehouse.fit(r1).well("B1").expected == F(10)


def test_publish_has_complete_provenance():
    design = PlateDesign("P", rows=3, cols=3, min_replicates=1, max_replicates=2)
    design.set_blank(["A1"])
    design.set_reference(["A2"], F(3))
    design.define_sample("S", ["B1"])
    design.block_wells(["C3"])
    layout = design.publish()
    assert len(layout.wells) == 9
    for well in layout.wells:
        assert well.name == Well(well.row, well.col).name
        assert well.provenance
        if well.role == "unused":
            assert well.expected is None and well.sample_id is None
        if well.role == "blank":
            assert well.expected == F(0)
    assert layout.well_map["C3"].blocked


def test_published_layout_roundtrip_and_digest():
    design = make_full_control_design()
    design.define_sample("S", ["A1", "B2"])
    layout = design.publish()
    payload = layout.to_dict()
    restored = PublishedLayout.from_dict(payload)
    assert restored.digest == layout.digest
    assert restored.to_dict() == payload


def test_published_layout_rejects_tampering():
    design = make_full_control_design()
    layout = design.publish()
    payload = layout.to_dict()
    reference_index = next(
        i for i, w in enumerate(payload["wells"]) if w["role"] == "reference"
    )
    payload["wells"][reference_index]["expected"] = "999"
    with pytest.raises(PublishedLayoutError):
        PublishedLayout.from_dict(payload)


# ---------------------------------------------------------------------------
# Readout import
# ---------------------------------------------------------------------------


def test_import_rejects_duplicate_measurements():
    design = PlateDesign("P", rows=2, cols=2, min_replicates=1, max_replicates=2)
    design.set_blank(["A1"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, [("A1", 1), ("A1", 2)])


def test_import_rejects_out_of_plate_well():
    design = PlateDesign("P", rows=2, cols=2, min_replicates=1, max_replicates=2)
    design.set_blank(["A1"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, {"A3": 1})
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, {"I1": 1})


def test_import_rejects_unused_well():
    design = PlateDesign("P", rows=2, cols=2, min_replicates=1, max_replicates=2)
    design.set_blank(["A1"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, {"B2": 1})


def test_import_rejects_blocked_well():
    design = PlateDesign("P", rows=2, cols=2, min_replicates=1, max_replicates=2)
    design.set_blank(["A1"])
    design.block_wells(["B2"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, {"B2": 1})


def test_import_rejects_positional_array_instead_of_zipping():
    design = PlateDesign("P", rows=2, cols=2)
    design.set_blank(["A1", "B1"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, [1.0, 2.0])


def test_import_accepts_record_and_pair_forms():
    design = PlateDesign("P", rows=2, cols=2)
    design.set_blank(["A1", "B1"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    warehouse.import_run(layout, [{"well": "A1", "value": "1/2"}, ("B1", "1/3")])
    run = warehouse.runs()[0]
    assert run.measurements["A1"] == F(1, 2)
    assert run.measurements["B1"] == F(1, 3)


def test_import_duplicate_run_id_rejected():
    design = PlateDesign("P", rows=2, cols=2)
    design.set_blank(["A1"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    warehouse.import_run(layout, {"A1": 1}, run_id="x")
    with pytest.raises(ImportError_):
        warehouse.import_run(layout, {"A1": 1}, run_id="x")


def test_plate_limit():
    warehouse = ReadoutWarehouse(max_plates=1)
    design = PlateDesign("P1", rows=2, cols=2)
    design.set_blank(["A1"])
    warehouse.bind_layout(design.publish())
    other = PlateDesign("P2", rows=2, cols=2)
    other.set_blank(["A1"])
    with pytest.raises(WarehouseError):
        warehouse.bind_layout(other.publish())


def test_external_layout_product_accepted():
    design = make_full_control_design("external")
    design.define_sample("S", ["A1", "B2"])
    payload = design.publish().to_dict()
    warehouse = ReadoutWarehouse()
    layout = warehouse.bind_layout(payload)
    assert layout.plate_id == "external"


# ---------------------------------------------------------------------------
# Fitting and identifiability
# ---------------------------------------------------------------------------


def test_full_control_recovery_exact():
    design = make_full_control_design()
    design.define_sample("S1", ["A1", "B2"])
    design.define_sample("S2", ["C3", "D4"])
    layout = design.publish()
    row_effects = {i: F(2 * i - 7, 2) for i in range(8)}
    col_effects = {j: F(2 * j - 11, 4) for j in range(12)}
    assert sum(row_effects.values()) == 0
    assert sum(col_effects.values()) == 0
    sample_values = {"S1": F(4), "S2": F(9, 2)}
    measurements = synth(layout, F(2, 5), row_effects, col_effects, sample_values)
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    assert report.rank == report.n_params
    assert report.nullity == 0
    assert report.parameter_unique
    assert report.effects["b"] == F(2, 5)
    assert report.residual_sum_of_squares == 0
    assert report.verification["ok"]
    for name, value in sample_values.items():
        summary = report.sample(name)
        assert summary.status == "COMPLETE"
        assert summary.mean == value
        assert summary.range == 0


def test_noisy_fit_minimises_squared_error():
    design = PlateDesign("P", rows=3, cols=4, min_replicates=1, max_replicates=2)
    for r in range(3):
        for c in range(4):
            design.set_reference([Well(r, c).name], F(10))
    design.define_sample("S", ["A4", "B4"])
    layout = design.publish()
    measurements = {w.name: F(10) for w in layout.wells if w.role != "unused"}
    measurements["C1"] = F(11)  # a control well is off by one
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    # The only error cannot be absorbed exactly: bias is additive and shared.
    assert report.residual_sum_of_squares > 0
    assert report.verification["ok"]
    # The optimum is characterised exactly by the normal equations.
    assert report.verification["normal_equations_zero"]


def test_disconnected_control_graph_marks_unidentifiable():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "A2", "B1", "B2"])
    design.set_reference(["C3", "C4", "D3", "D4"], F(6))
    design.define_sample("S-far", ["E7", "F8"])
    layout = design.publish()
    measurements = synth(
        layout,
        F(1, 3),
        {i: F(2 * i - 7, 2) for i in range(8)},
        {j: F(3 * j - 18, 4) for j in range(12)},
        {"S-far": F(8)},
    )
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    assert report.nullity > 0
    assert not report.parameter_unique
    assert report.sample("S-far").status == "INCOMPLETE"
    for result in report.wells:
        if result.role == "sample":
            assert result.status == "UNIDENTIFIABLE"
            assert result.proof is not None


def test_unique_bias_with_nonunique_parameters():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "A2"])
    design.set_reference(["B1"], F(5))
    design.define_sample("S-corner", ["B2"])
    design.define_sample("S-far", ["D4", "E5"])
    layout = design.publish()
    # Existing rows {0,1,3,4} and columns {0,1,3,4}; zero-sum effects.
    row_effects = {0: F(1), 1: F(-1), 3: F(1, 2), 4: F(-1, 2)}
    col_effects = {0: F(1, 4), 1: F(-1, 4), 3: F(1, 4), 4: F(-1, 4)}
    assert sum(row_effects.values()) == 0
    assert sum(col_effects.values()) == 0
    b = F(1, 2)
    measurements = {}
    for well in layout.wells:
        if well.role == "unused":
            continue
        bias = b + row_effects[well.row] + col_effects[well.col]
        if well.role == "blank":
            measurements[well.name] = bias
        elif well.role == "reference":
            measurements[well.name] = well.expected + bias
        elif well.sample_id == "S-corner":
            measurements[well.name] = F(4) + bias
        else:
            measurements[well.name] = F(8) + bias
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    assert report.nullity > 0
    assert not report.parameter_unique
    corner = report.well("B2")
    assert corner.bias_identifiable
    assert corner.status == "CORRECTED"
    assert corner.corrected == F(4)
    assert report.well("D4").status == "UNIDENTIFIABLE"
    assert report.sample("S-corner").status == "COMPLETE"
    assert report.sample("S-far").status == "INCOMPLETE"
    assert report.verification["identifiability_claims_verified"]


def test_unidentifiability_proof_is_a_real_null_direction():
    design = PlateDesign("P", rows=8, cols=12, min_replicates=1, max_replicates=2)
    design.set_blank(["A1", "A2"])
    design.set_reference(["B1"], F(5))
    design.define_sample("S-far", ["D4", "E5"])
    layout = design.publish()
    measurements = synth(
        layout,
        F(1, 2),
        {0: F(-1), 1: F(1), 2: F(0), 3: F(0), 4: F(0)},
        {j: F(0) for j in range(12)},
        {"S-far": F(8)},
    )
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    factor = _factor_for(layout, report.control_wells)
    structure = _structure_for(layout)
    for result in report.wells:
        if result.status != "UNIDENTIFIABLE":
            continue
        proof = result.proof
        assert proof is not None
        # Reconstruct the parameter direction from the recorded effect deltas
        # and confirm it changes this well's bias by exactly 1 and no residual.
        direction = [F(0) for _ in range(structure.n_params)]
        direction[0] = F(proof["delta_b"])
        for index, row in enumerate(structure.rows[:-1]):
            direction[1 + index] = F(proof["delta_rows"][str(row)])
        col_base = 1 + structure.row_coords
        for index, col in enumerate(structure.cols[:-1]):
            direction[col_base + index] = F(proof["delta_cols"][str(col)])
        from plate_effects.warehouse import _feature
        from plate_effects.rational import dot

        feature = _feature(structure, result.row, result.col)
        assert dot(feature, direction) == 1
        for row_vec in factor.design:
            assert dot(row_vec, direction) == 0


def test_missing_replicate_marks_incomplete_with_reason():
    design = make_full_control_design("P-missing-rep", min_replicates=2, max_replicates=2)
    design.define_sample("S", ["A1", "B1"])
    layout = design.publish()
    measurements = synth(
        layout,
        F(0),
        {i: F(0) for i in range(8)},
        {j: F(0) for j in range(12)},
        {"S": F(3)},
    )
    del measurements["B1"]
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    assert report.well("B1").status == "MISSING"
    summary = report.sample("S")
    assert summary.status == "INCOMPLETE"
    assert summary.mean is None
    assert [r["well"] for r in summary.reasons] == ["B1"]
    assert report.missing_wells == ("B1",)
    # The observed replicate is still corrected; it is not silently discarded.
    assert report.well("A1").status == "CORRECTED"
    assert report.well("A1").corrected == F(3)


def test_missing_control_wells_are_reported():
    design = make_full_control_design("P-missing-control", min_replicates=1, max_replicates=2)
    design.define_sample("S", ["A1", "B1"])
    layout = design.publish()
    measurements = {
        w.name: (F(0) if w.role == "blank" else w.expected)
        for w in layout.wells
        if w.role in ("blank", "reference")
    }
    measurements["A1"] = F(0)  # sample well A1 carries an arbitrary observation
    del measurements["H12"]
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    assert "H12" in report.missing_control_wells


def test_corrected_mean_and_range_exact():
    design = make_full_control_design()
    design.define_sample("S", ["A1", "B2", "C3"])
    layout = design.publish()
    measurements = synth(
        layout,
        F(1, 3),
        {i: F(2 * i - 7, 2) for i in range(8)},
        {j: F(3 * j - 18, 4) for j in range(12)},
        {"S": F(7)},
    )
    measurements["A1"] += F(1, 3)
    warehouse = ReadoutWarehouse()
    summary = warehouse.fit(warehouse.import_run(layout, measurements)).sample("S")
    assert summary.status == "COMPLETE"
    assert summary.mean == F(7) + F(1, 9)
    assert summary.range == F(1, 3)


def test_no_cross_plate_pooling():
    warehouse = ReadoutWarehouse()
    reports = []
    for plate in range(2):
        design = make_full_control_design(f"P{plate}")
        design.define_sample("S", ["A1", "B2"])
        layout = design.publish()
        measurements = synth(
            layout,
            F(1 + plate),
            {i: F(0) for i in range(8)},
            {j: F(0) for j in range(12)},
            {"S": F(3)},
        )
        reports.append(warehouse.fit(warehouse.import_run(layout, measurements)))
    assert reports[0].effects["b"] == F(1)
    assert reports[1].effects["b"] == F(2)


def test_factor_cache_is_reused_and_correct():
    clear_caches()
    design = make_full_control_design()
    design.define_sample("S", ["A1", "B2"])
    layout = design.publish()
    warehouse = ReadoutWarehouse()
    first = warehouse.fit(warehouse.import_run(layout, synth(
        layout, F(0), {i: F(0) for i in range(8)}, {j: F(0) for j in range(12)}, {"S": F(1)}
    )))
    a = _factor_for(layout, first.control_wells)
    b = _factor_for(layout, first.control_wells)
    assert a is b


def test_numerical_rank_crosscheck_agrees_on_exact_design():
    design = make_full_control_design()
    design.define_sample("S", ["A1", "B2"])
    layout = design.publish()
    measurements = synth(
        layout, F(0), {i: F(0) for i in range(8)}, {j: F(0) for j in range(12)}, {"S": F(1)}
    )
    warehouse = ReadoutWarehouse()
    report = warehouse.fit(warehouse.import_run(layout, measurements))
    assert report.numerical_rank == report.rank
    assert report.rank_tolerance > 0
