"""Tests for the glyph font builder and the offline label interpreter.

The exhaustive tests cross-check the capped dynamic program against a brute
force enumeration of every digit string, proving that the two-witness cap
never changes the no-solution / unique / ambiguous answer.
"""

from __future__ import annotations

import itertools
import json
from fractions import Fraction as F

import pytest

from plate_effects.font_builder import (
    CorrectionRun,
    FontError,
    GlyphFont,
    SampleSheet,
    build_font,
    sheet_from_rectangle,
)
from plate_effects.label_reader import (
    LabelContract,
    LabelInterpreter,
    LabelReaderError,
    STATUS_AMBIGUOUS,
    STATUS_NO_SOLUTION,
    STATUS_UNIQUE,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _well(run_id: str, r: int, c: int) -> str:
    return f"{run_id}-r{r}c{c}"


def make_run(run_id: str, grid: list[list[object]]) -> CorrectionRun:
    values = {
        _well(run_id, r, c): value
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
    }
    return CorrectionRun.from_values(run_id, values)


def make_sheet(sheet_id: str, digit: str, run: CorrectionRun, h: int, w: int, lo, hi) -> SampleSheet:
    wells = [_well(run.run_id, r, c) for r in range(h) for c in range(w)]
    return sheet_from_rectangle(sheet_id, digit, run, height=h, width=w, wells=wells, t_lo=lo, t_hi=hi)


def canonical_font() -> GlyphFont:
    run_one = make_run("run-one", [[1]])
    run_two = make_run("run-two", [[2, 3]])
    return build_font(
        [run_one, run_two],
        [
            make_sheet("s1", "1", run_one, 1, 1, F(0), F(1)),
            make_sheet("s2", "2", run_two, 1, 2, F(2), F(4)),
        ],
        font_id="canonical",
    )


def brute_force_solutions(
    font: GlyphFont, band: list[list[int]], k: int, contract: LabelContract
) -> list[str]:
    """Enumerate every digit string and every font variant per position."""

    reader = LabelInterpreter(font)
    digits = sorted(reader._digits)
    solutions: set[str] = set()
    for combo in itertools.product(digits, repeat=k):
        widths = [reader._digits[d][0] for d in combo]
        if sum(widths) != len(band):
            continue
        position = 0
        ok = True
        for digit in combo:
            width, variants = reader._digits[digit]
            window = tuple(tuple(band[position + off]) for off in range(width))
            if not any(variant.columns == window for variant in variants):
                ok = False
                break
            position += width
        if not ok:
            continue
        checksum = sum(contract.weights[i] * int(combo[i]) for i in range(k)) % contract.modulus
        if checksum in contract.allowed_remainders:
            solutions.add("".join(combo))
    return sorted(solutions)


# ---------------------------------------------------------------------------
# Font construction: whole-exposure threshold semantics
# ---------------------------------------------------------------------------


def test_canonical_sheet_produces_exactly_three_patterns():
    font = canonical_font()
    patterns = {tuple(col[0] for col in v.columns) for v in font.variants["2"]}
    assert patterns == {(1, 1), (0, 1), (0, 0)}
    # The per-cell patchwork 10 is structurally impossible.
    assert (1, 0) not in patterns


def test_every_variant_traces_to_one_whole_sheet_and_threshold():
    font = canonical_font()
    for variant in font.variants["2"]:
        assert len(variant.origins) >= 1
        for origin in variant.origins:
            assert origin.sheet_id == "s2"
            assert F(2) <= origin.threshold <= F(4)
            assert len(origin.cells) == 2  # whole bitmap, never single columns
            # Re-apply the shared threshold: it must reproduce the bitmap.
            for cell in origin.cells:
                expected = 1 if cell.corrected >= origin.threshold else 0
                assert variant.columns[cell.col][cell.row] == expected


def test_threshold_representatives_cover_endpoints_and_crossings():
    # Values 2, 3 on [2, 4]: representatives 2, 3, 4 yield 11, 01, 00.
    run = make_run("t-run", [[2, 3]])
    sheet = make_sheet("t", "2", run, 1, 2, F(2), F(4))
    font = build_font([run], [sheet])
    thresholds = {origin.threshold for v in font.variants["2"] for origin in v.origins}
    assert thresholds == {F(2), F(3), F(4)}


def test_identical_bitmap_at_two_thresholds_merges_into_one_variant():
    run = make_run("m-run", [[1, 5]])
    sheet = make_sheet("m", "2", run, 1, 2, F(0), F(6))
    font = build_font([run], [sheet])
    patterns = sorted(tuple(col[0] for col in v.columns) for v in font.variants["2"])
    assert patterns == [(0, 0), (0, 1), (1, 1)]
    variant_11 = next(v for v in font.variants["2"] if tuple(col[0] for col in v.columns) == (1, 1))
    assert len(variant_11.origins) == 1


def test_duplicate_sheets_add_provenance_not_variants():
    run_a = make_run("a", [[2, 3]])
    run_b = make_run("b", [[2, 3]])
    font = build_font(
        [run_a, run_b],
        [
            make_sheet("a2", "2", run_a, 1, 2, F(2), F(4)),
            make_sheet("b2", "2", run_b, 1, 2, F(2), F(4)),
        ],
    )
    for variant in font.variants["2"]:
        assert {o.sheet_id for o in variant.origins} == {"a2", "b2"}
    assert len(font.variants["2"]) == 3


def test_missing_cell_invalidates_sheet_without_painting_dark():
    good = make_run("good", [[1, 1], [0, 1], [1, 1]])
    broken = make_run("broken", [[1, 1], [None, 1], [1, 1]])
    font = build_font(
        [good, broken],
        [
            make_sheet("g", "3", good, 3, 2, F(1), F(1)),
            make_sheet("b", "3", broken, 3, 2, F(1), F(1)),
        ],
    )
    record = next(r for r in font.invalid_sheets if r.sheet_id == "b")
    assert record.reason == "MISSING_CELL"
    assert record.cells == ((1, 0),)
    assert len(font.variants["3"]) == 1


def test_unidentifiable_hole_is_recorded_separately():
    from plate_effects.font_builder import PlateHole

    run = CorrectionRun(
        "u-run",
        {"u-run-r0c0": PlateHole("u-run", "u-run-r0c0", None, "UNIDENTIFIABLE")},
    )
    sheet = sheet_from_rectangle(
        "u", "5", run, height=1, width=1, wells=["u-run-r0c0"], t_lo=F(0), t_hi=F(1)
    )
    font = build_font([run], [sheet])
    assert "5" in font.missing_digits
    assert font.invalid_sheets[0].reason == "UNRECOGNISABLE_CELL"


def test_digit_without_valid_sheet_is_visible_in_export_and_round_trips(tmp_path):
    run = make_run("r", [[1]])
    font = build_font([run], [make_sheet("s", "1", run, 1, 1, F(1), F(1))])
    assert set(font.missing_digits) == set("023456789")
    payload = font.to_dict()
    assert payload["digits"]["7"] == []
    path = tmp_path / "font.json"
    font.save(str(path))
    reloaded = GlyphFont.load(str(path))
    assert reloaded.digest == font.digest
    assert reloaded.missing_digits == font.missing_digits
    # Tampered content breaks the digest.
    damaged = json.loads(path.read_text(encoding="utf-8"))
    damaged["height"] = 7
    with pytest.raises(FontError):
        GlyphFont.from_dict(damaged)


def test_font_rejects_conflicting_widths_for_one_digit():
    run_a = make_run("a", [[1]])
    run_b = make_run("b", [[1, 1]])
    with pytest.raises(FontError):
        build_font(
            [run_a, run_b],
            [
                make_sheet("a1", "1", run_a, 1, 1, F(1), F(1)),
                make_sheet("b1", "1", run_b, 1, 2, F(1), F(1)),
            ],
        )


def test_font_rejects_mixed_heights_and_bad_bounds():
    run = make_run("r", [[1]])
    with pytest.raises(FontError):
        SampleSheet("s", "1", "r", 9, 1, {(r, c): "x" for r in range(9) for c in range(1)}, F(0), F(1))
    with pytest.raises(FontError):
        make_sheet("s", "1", run, 1, 1, F(5), F(2))


def test_sheet_requires_complete_rectangle_mapping():
    with pytest.raises(FontError):
        SampleSheet(
            "s", "1", "r", 1, 2, {(0, 0): "r-r0c0"}, F(0), F(1)
        )


# ---------------------------------------------------------------------------
# Interpreter: canonical acceptance examples
# ---------------------------------------------------------------------------


def test_canonical_band_remainder_one_gives_12():
    font = canonical_font()
    reader = LabelInterpreter(font)
    band = [[1], [1], [1]]
    result = reader.solve(band, 2, LabelContract.create((2, 1), 3, [1]))
    assert result.status == STATUS_UNIQUE
    assert result.digits == ("12",)
    witness = result.witnesses[0]
    assert [(c.column_start, c.column_end) for c in witness.characters] == [(0, 1), (1, 3)]
    assert witness.checksum == 1


def test_canonical_band_remainder_two_gives_21():
    reader = LabelInterpreter(canonical_font())
    result = reader.solve([[1], [1], [1]], 2, LabelContract.create((2, 1), 3, [2]))
    assert result.status == STATUS_UNIQUE and result.digits == ("21",)


def test_canonical_band_both_remainders_is_ambiguous_with_two_witnesses():
    reader = LabelInterpreter(canonical_font())
    result = reader.solve([[1], [1], [1]], 2, LabelContract.create((2, 1), 3, [1, 2]))
    assert result.status == STATUS_AMBIGUOUS
    assert set(result.digits) == {"12", "21"}
    for witness in result.witnesses:
        intervals = [(c.column_start, c.column_end) for c in witness.characters]
        assert intervals[0][0] == 0 and intervals[-1][1] == 3


def test_corrected_font_with_only_10_makes_band_unsolvable():
    run_one = make_run("run-one", [[1]])
    run_fixed = make_run("run-fixed", [[3, 2]])
    font = build_font(
        [run_one, run_fixed],
        [
            make_sheet("s1", "1", run_one, 1, 1, F(0), F(1)),
            make_sheet("s2", "2", run_fixed, 1, 2, F(5, 2), F(5, 2)),
        ],
    )
    assert {tuple(c[0] for c in v.columns) for v in font.variants["2"]} == {(1, 0)}
    result = LabelInterpreter(font).solve(
        [[1], [1], [1]], 2, LabelContract.create((2, 1), 3, [1])
    )
    assert result.status == STATUS_NO_SOLUTION


def test_band_must_be_consumed_from_first_to_last_column():
    reader = LabelInterpreter(canonical_font())
    # Four columns cannot be exactly two glyphs (widths 1 + 2 = 3).
    result = reader.solve([[1], [1], [1], [0]], 2, LabelContract.create((2, 1), 3, [1, 2]))
    assert result.status == STATUS_NO_SOLUTION
    # One column cannot be two glyphs either.
    result_short = reader.solve([[1]], 2, LabelContract.create((2, 1), 3, [1, 2]))
    assert result_short.status == STATUS_NO_SOLUTION


def test_leading_zero_is_a_character():
    runs, sheets = [], []
    for digit, grid in {
        "0": [[1, 1], [1, 1]],
        "1": [[0, 1], [0, 1]],
    }.items():
        run = make_run(f"r{digit}", grid)
        runs.append(run)
        sheets.append(make_sheet(f"s{digit}", digit, run, 2, 2, F(1), F(1)))
    font = build_font(runs, sheets)
    reader = LabelInterpreter(font)
    band = [list(col) for variant in (font.variants["0"][0], font.variants["1"][0]) for col in variant.columns]
    result = reader.solve(band, 2, LabelContract.create((10, 1), 7, [1]))  # 0*10 + 1
    assert result.status == STATUS_UNIQUE and result.digits == ("01",)


def test_contract_swap_re_reads_same_band_without_rebuilding():
    reader = LabelInterpreter(canonical_font())
    band = [[1], [1], [1]]
    ok = reader.solve(band, 2, LabelContract.create((2, 1), 3, [1]))
    dead = reader.solve(band, 2, LabelContract.create((2, 1), 3, [0]))
    assert ok.status == STATUS_UNIQUE and dead.status == STATUS_NO_SOLUTION


# ---------------------------------------------------------------------------
# Exhaustive cross-checks against brute force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_dp_matches_brute_force_on_random_fonts_and_bands(seed):
    import random

    rng = random.Random(seed)
    height = rng.randint(1, 3)
    runs, sheets, present_digits = [], [], []
    for digit in "01234":
        if rng.random() < 0.25:
            continue  # missing digit
        width = rng.randint(1, 3)
        grid = [[rng.randint(0, 4) for _ in range(width)] for _ in range(height)]
        run = make_run(f"run-{digit}-{seed}", grid)
        runs.append(run)
        lo = rng.randint(0, 3)
        hi = lo + rng.randint(0, 3)
        sheets.append(make_sheet(f"sheet-{digit}", digit, run, height, width, F(lo), F(hi)))
        present_digits.append(digit)
    if not present_digits:
        return
    font = build_font(runs, sheets)
    reader = LabelInterpreter(font)

    k = rng.randint(1, 4)
    target = "".join(rng.choice(present_digits) for _ in range(k))
    band: list[list[int]] = []
    for digit in target:
        variant = rng.choice(font.variants[digit])
        band.extend(list(col) for col in variant.columns)

    modulus = rng.choice((2, 3, 5, 7))
    weights = tuple(rng.randint(-3, 3) for _ in range(k))
    all_remainders = list(range(modulus))
    chosen = frozenset(rng.sample(all_remainders, rng.randint(1, modulus)))
    contract = LabelContract.create(weights, modulus, chosen)

    expected = brute_force_solutions(font, band, k, contract)
    result = reader.solve(band, k, contract)
    if not expected:
        assert result.status == STATUS_NO_SOLUTION
    elif len(expected) == 1:
        assert result.status == STATUS_UNIQUE and result.digits == (expected[0],)
    else:
        assert result.status == STATUS_AMBIGUOUS
        assert len(result.digits) == 2 and result.digits[0] != result.digits[1]
        assert set(result.digits) <= set(expected)


def test_identical_column_run_does_not_duplicate_one_string():
    # All digits share one identical six-column bitmap: every cut of the band
    # gives the same family of strings; duplicates must collapse exactly.
    runs, sheets = [], []
    for digit in "0123456789":
        run = make_run(f"r{digit}", [[1] * 6, [0] * 6])
        runs.append(run)
        sheets.append(make_sheet(f"s{digit}", digit, run, 2, 6, F(1), F(1)))
    font = build_font(runs, sheets)
    reader = LabelInterpreter(font)
    band = [[1, 0] for _ in range(60)]  # 10 digits x 6 columns
    contract = LabelContract.create(tuple([1] * 10), 97, [0])
    result = reader.solve(band, 10, contract)
    assert result.status == STATUS_UNIQUE
    assert result.digits == ("0" * 10,)


def test_large_band_80_digits_480_columns_stays_bounded():
    runs, sheets = [], []
    for digit in "0123456789":
        run = make_run(f"r{digit}", [[1] * 6, [0] * 6])
        runs.append(run)
        sheets.append(make_sheet(f"s{digit}", digit, run, 2, 6, F(1), F(1)))
    font = build_font(runs, sheets)
    reader = LabelInterpreter(font)
    band = [[1, 0] for _ in range(480)]
    ambiguous = reader.solve(
        band, 80, LabelContract.create(tuple([1] * 80), 97, [0, 1])
    )
    assert ambiguous.status == STATUS_AMBIGUOUS
    assert ambiguous.witnesses[0].digits != ambiguous.witnesses[1].digits
    assert {sum(int(d) for d in w.digits) for w in ambiguous.witnesses} == {0, 1}


# ---------------------------------------------------------------------------
# Validation and limits
# ---------------------------------------------------------------------------


def test_contract_validation():
    with pytest.raises(LabelReaderError):
        LabelContract.create((), 3, [0])
    with pytest.raises(LabelReaderError):
        LabelContract.create((1,), 98, [0])
    with pytest.raises(LabelReaderError):
        LabelContract.create((1,), 3, [3])
    with pytest.raises(LabelReaderError):
        LabelContract.create((1,), 3, [])


def test_reader_rejects_wrong_height_and_non_binary_band():
    reader = LabelInterpreter(canonical_font())
    with pytest.raises(LabelReaderError):
        reader.solve([[1, 0]], 1, LabelContract.create((1,), 3, [0]))
    with pytest.raises(LabelReaderError):
        reader.solve([[2]], 1, LabelContract.create((1,), 3, [0]))


def test_weight_count_must_match_k():
    reader = LabelInterpreter(canonical_font())
    with pytest.raises(LabelReaderError):
        reader.solve([[1]], 2, LabelContract.create((1,), 3, [0]))


def test_font_json_is_self_contained_for_offline_reader(tmp_path):
    font = canonical_font()
    path = tmp_path / "offline.json"
    font.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "glyph-font/v1"
    reader = LabelInterpreter.from_file(str(path))
    result = reader.solve([[1], [1], [1]], 2, LabelContract.create((2, 1), 3, [1]))
    assert result.digits == ("12",)
    # Witness origin carries the full verifiable source.
    origin = result.witnesses[0].characters[1].origin
    assert origin.run_id == "run-two"
    assert {cell.well for cell in origin.cells} == {"run-two-r0c0", "run-two-r0c1"}


def test_max_geometry_8_by_12_and_fractional_thresholds():
    cells, values = {}, {}
    for r in range(8):
        for c in range(12):
            well = f"w{r}_{c}"
            cells[(r, c)] = well
            values[well] = F(1, 2)
    run = CorrectionRun.from_values("big", values)
    sheet = SampleSheet("s", "8", "big", 8, 12, cells, F(1, 3), F(2, 3))
    font = build_font([run], [sheet])
    assert len(font.variants["8"]) == 2
    thresholds = {v.origins[0].threshold for v in font.variants["8"]}
    assert thresholds == {F(1, 3), F(2, 3)}
    for variant in font.variants["8"]:
        assert variant.width == 12 and variant.height == 8
        assert len(variant.origins[0].cells) == 96
    reader = LabelInterpreter(font)
    lit_band = [[1] * 8 for _ in range(12)]
    result = reader.solve(lit_band, 1, LabelContract.create((1,), 5, [3]))
    assert result.status == "unique" and result.digits == ("8",)


def test_exact_fractional_crossing_points():
    run = CorrectionRun.from_values("frac", {"a": F(1, 2), "b": F(1, 4)})
    sheet = SampleSheet(
        "f", "9", "frac", 1, 2, {(0, 0): "a", (0, 1): "b"}, F(1, 4), F(1, 2)
    )
    font = build_font([run], [sheet])
    patterns = sorted(tuple(col[0] for col in v.columns) for v in font.variants["9"])
    # t = 1/4 -> 11 (both cells >= 1/4); t = 1/2 -> 10.
    assert patterns == [(1, 0), (1, 1)]


def test_sheet_above_max_dimensions_is_rejected():
    with pytest.raises(FontError):
        SampleSheet(
            "s",
            "1",
            "r",
            9,
            1,
            {(r, 0): f"w{r}" for r in range(9)},
            F(0),
            F(1),
        )
    with pytest.raises(FontError):
        SampleSheet(
            "s",
            "1",
            "r",
            1,
            13,
            {(0, c): f"w{c}" for c in range(13)},
            F(0),
            F(1),
        )
