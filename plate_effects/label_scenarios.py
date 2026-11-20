"""End-to-end demonstrations of font construction and offline band reading.

Run with::

    python -m plate_effects.label_scenarios <outdir>

Each scenario builds synthetic correction *runs*, turns sample sheets into an
exportable glyph font, saves the font to disk and re-opens it in a fresh
interpreter that has no access to the runs.  Every assertion below mirrors a
requirement of the task statement.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction as F

from .font_builder import (
    CorrectionRun,
    GlyphFont,
    SampleSheet,
    build_font,
    sheet_from_rectangle,
)
from .label_reader import (
    LabelContract,
    LabelInterpreter,
    LabelReaderError,
)

F0 = F(0)
F1 = F(1)


def _dump(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _well(prefix: str, r: int, c: int) -> str:
    return f"{prefix}-r{r}c{c}"


def _run_from_grid(run_id: str, grid: list[list[object]]) -> CorrectionRun:
    """A run whose well names encode their (row, col) source coordinates."""

    values = {
        _well(run_id, r, c): value
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
    }
    return CorrectionRun.from_values(run_id, values)


def _sheet(
    sheet_id: str,
    digit: str,
    run: CorrectionRun,
    height: int,
    width: int,
    t_lo: object,
    t_hi: object,
) -> SampleSheet:
    wells = [_well(run.run_id, r, c) for r in range(height) for c in range(width)]
    return sheet_from_rectangle(
        sheet_id,
        digit,
        run,
        height=height,
        width=width,
        wells=wells,
        t_lo=t_lo,
        t_hi=t_hi,
    )


def _band_from_digits(font: GlyphFont, digits: str) -> list[list[int]]:
    """Concatenate each digit's first variant: a clean, unseparated band."""

    band: list[list[int]] = []
    for digit in digits:
        variant = font.variants[digit][0]
        band.extend(list(column) for column in variant.columns)
    return band


# ---------------------------------------------------------------------------
# Scenario 1: the task's canonical 1/2 example (height 1)
# ---------------------------------------------------------------------------


def scenario_canonical_one_two(outdir: str) -> dict:
    # Sheet "2" reads 2, 3 across its two cells; the closed exposure interval
    # [2, 4] must produce exactly 11, 01, 00 and never the patchwork 10.
    run_two = _run_from_grid("run-two", [[2, 3]])
    run_one = _run_from_grid("run-one", [[1]])
    sheets = [
        _sheet("sheet-1", "1", run_one, 1, 1, F0, F1),
        _sheet("sheet-2", "2", run_two, 1, 2, F(2), F(4)),
    ]
    font = build_font([run_one, run_two], sheets, font_id="canonical-font")
    path = os.path.join(outdir, "font_canonical.json")
    font.save(path)
    reopened = LabelInterpreter.from_file(path)  # runs are not needed anymore

    two_variants = {v.columns: v for v in font.variants["2"]}
    produced = {tuple(c[0] for c in columns): columns for columns in two_variants}
    assert set(produced) == {(1, 1), (0, 1), (0, 0)}, produced
    assert (1, 0) not in produced, "patchwork column 10 must never be generated"
    # Each distinct bitmap traces back to one whole sheet at one threshold.
    for variant in font.variants["2"]:
        assert len(variant.origins) == 1
        origin = variant.origins[0]
        assert origin.sheet_id == "sheet-2"
        assert F(2) <= origin.threshold <= F(4)
        assert len(origin.cells) == 2  # whole bitmap, not per-column evidence

    band = [[1], [1], [1]]  # scan 111, k = 2
    contract_r1 = LabelContract.create((2, 1), 3, [1], name="remainder-1")
    contract_r2 = LabelContract.create((2, 1), 3, [2], name="remainder-2")
    contract_both = LabelContract.create((2, 1), 3, [1, 2], name="remainders-1-2")

    answer_r1 = reopened.solve(band, 2, contract_r1)
    answer_r2 = reopened.solve(band, 2, contract_r2)
    answer_both = reopened.solve(band, 2, contract_both)
    assert answer_r1.status == "unique" and answer_r1.digits == ("12",)
    assert answer_r2.status == "unique" and answer_r2.digits == ("21",)
    assert answer_both.status == "ambiguous"
    assert set(answer_both.digits) == {"12", "21"}
    # Witnesses carry per-character column intervals spanning the whole band.
    for witness in answer_both.witnesses:
        intervals = [(c.column_start, c.column_end) for c in witness.characters]
        assert intervals == [(0, 1), (1, 3)] or intervals == [(0, 2), (2, 3)]

    # Corrected sheet for 2 whose only legal bitmap is 10: the corrected
    # exposure interval is the single threshold 5/2 (between values 2 and 3).
    # The original band 111 then has no legal decoding under remainder 1.
    run_two_fixed = _run_from_grid("run-two-fixed", [[3, 2]])
    font_fixed = build_font(
        [run_one, run_two_fixed],
        [
            _sheet("sheet-1", "1", run_one, 1, 1, F0, F1),
            _sheet("sheet-2f", "2", run_two_fixed, 1, 2, F(5, 2), F(5, 2)),
        ],
        font_id="fixed-font",
    )
    fixed_patterns = {tuple(c[0] for c in v.columns) for v in font_fixed.variants["2"]}
    assert fixed_patterns == {(1, 0)}, fixed_patterns
    fixed_reader = LabelInterpreter(font_fixed)
    no_solution = fixed_reader.solve(band, 2, contract_r1)
    assert no_solution.status == "no_solution"

    return {
        "name": "canonical_one_two",
        "font_path": path,
        "digit_2_patterns": ["".join(str(b) for b in pattern) for pattern in sorted(produced)],
        "forbidden_pattern": "10",
        "read_remainder_1": answer_r1.to_dict(),
        "read_remainder_2": answer_r2.to_dict(),
        "read_both_remainders": answer_both.to_dict(),
        "corrected_2_only_10": no_solution.to_dict(),
    }


# ---------------------------------------------------------------------------
# Scenario 2: duplicate sheets and duplicate thresholds create no ambiguity
# ---------------------------------------------------------------------------


def scenario_duplicate_sheets(outdir: str) -> dict:
    run_a = _run_from_grid("dup-run-a", [[2, 3]])
    run_b = _run_from_grid("dup-run-b", [[2, 3]])  # identical corrected values
    sheets = [
        _sheet("dup-1a", "1", run_a, 1, 1, F0, F1),
        _sheet("dup-2a", "2", run_a, 1, 2, F(2), F(4)),
        # Second exposure of the same digit with the same grid and interval:
        # it adds provenance, never a new bitmap.
        _sheet("dup-2b", "2", run_b, 1, 2, F(2), F(4)),
    ]
    font = build_font([run_a, run_b], sheets, font_id="dup-font")
    path = os.path.join(outdir, "font_duplicate.json")
    font.save(path)
    reader = LabelInterpreter.from_file(path)

    patterns = {tuple(c[0] for c in v.columns) for v in font.variants["2"]}
    assert patterns == {(1, 1), (0, 1), (0, 0)}
    for variant in font.variants["2"]:
        # The same bitmap is now witnessed by two independent sheets.
        assert {o.sheet_id for o in variant.origins} == {"dup-2a", "dup-2b"}

    band = [[1], [1], [1]]
    contract = LabelContract.create((2, 1), 3, [1, 2], name="dup-contract")
    result = reader.solve(band, 2, contract)
    # Two distinct digit strings remain ambiguous, but each is reported once.
    assert result.status == "ambiguous"
    assert result.digits == ("12", "21")
    for witness in result.witnesses:
        # Multiple cuts of identical columns never duplicate the string.
        assert all(len(c.origin.cells) == 2 or c.digit == "1" for c in witness.characters)
    return {"name": "duplicate_sheets", "font_path": path, "read": result.to_dict()}


# ---------------------------------------------------------------------------
# Scenario 3: one bitmap at several thresholds is still one glyph
# ---------------------------------------------------------------------------


def scenario_threshold_representatives(outdir: str) -> dict:
    # Values 1 and 5 inside [0, 6]: representatives are 0, 1, 5, 6 but the
    # bitmaps at t=0 and t=1 are identical (11); they must merge.
    run = _run_from_grid("thresh-run", [[1, 5]])
    sheet = _sheet("thresh-2", "2", run, 1, 2, F(0), F(6))
    font = build_font([run], [sheet], font_id="thresh-font")
    patterns = sorted(tuple(c[0] for c in v.columns) for v in font.variants["2"])
    assert patterns == [(0, 0), (0, 1), (1, 1)], patterns
    variant_11 = next(v for v in font.variants["2"] if tuple(c[0] for c in v.columns) == (1, 1))
    # One origin per (sheet, bitmap); the chosen threshold is a member of the
    # closed interval where that bitmap really occurs.
    assert len(variant_11.origins) == 1
    assert variant_11.origins[0].threshold in (F(0), F(1))
    path = os.path.join(outdir, "font_thresholds.json")
    font.save(path)
    return {
        "name": "threshold_representatives",
        "font_path": path,
        "patterns": ["".join(str(b) for b in pattern) for pattern in patterns],
        "representative_threshold": str(variant_11.origins[0].threshold),
    }


# ---------------------------------------------------------------------------
# Scenario 4: missing cells are not painted dark
# ---------------------------------------------------------------------------


def scenario_missing_cells(outdir: str) -> dict:
    # Digit 3: one valid sheet plus one sheet with an unmeasured hole.
    good_grid: list[list[object]] = [
        [1, 1],
        [0, 1],
        [1, 1],
    ]
    broken_grid: list[list[object]] = [
        [1, 1],
        [None, 1],  # MISSING cell
        [1, 1],
    ]
    run_good = _run_from_grid("miss-run-good", good_grid)
    run_broken = _run_from_grid("miss-run-broken", broken_grid)
    sheets = [
        _sheet("three-good", "3", run_good, 3, 2, F(1), F(1)),
        _sheet("three-broken", "3", run_broken, 3, 2, F(1), F(1)),
    ]
    font = build_font([run_good, run_broken], sheets, font_id="missing-font")
    invalid = {record.sheet_id: record for record in font.invalid_sheets}
    assert "three-broken" in invalid
    assert invalid["three-broken"].reason == "MISSING_CELL"
    assert invalid["three-broken"].cells == ((1, 0),)
    # The good sheet still yields exactly one bitmap (threshold fixed at 1).
    assert len(font.variants["3"]) == 1
    assert font.variants["3"][0].columns == ((1, 0, 1), (1, 1, 1))
    path = os.path.join(outdir, "font_missing.json")
    font.save(path)
    return {
        "name": "missing_cells",
        "font_path": path,
        "invalid_sheets": [record.to_dict() for record in font.invalid_sheets],
        "three_variants": len(font.variants["3"]),
    }


# ---------------------------------------------------------------------------
# Scenario 5: a digit with no valid sheet is visible in the export
# ---------------------------------------------------------------------------


def scenario_digit_without_evidence(outdir: str) -> dict:
    run = _run_from_grid("noev-run", [[1, 1]])
    # Only digit 1 is registered; digit 4 has one sheet with a missing cell.
    broken = CorrectionRun.from_values("noev-broken", {"noev-broken-r0c0": None})
    sheets = [
        _sheet("one", "1", run, 1, 1, F(1), F(1)),
        sheet_from_rectangle(
            "four-broken",
            "4",
            broken,
            height=1,
            width=1,
            wells=["noev-broken-r0c0"],
            t_lo=F(1),
            t_hi=F(1),
        ),
    ]
    font = build_font([run, broken], sheets, font_id="no-evidence-font")
    assert "4" in font.missing_digits
    assert "4" not in font.variants
    payload = font.to_dict()
    assert payload["digits"]["4"] == []
    assert payload["missing_digits"] == [d for d in "0123456789" if d != "1"]
    path = os.path.join(outdir, "font_no_evidence.json")
    font.save(path)
    reloaded = GlyphFont.load(path)
    assert reloaded.missing_digits == font.missing_digits
    return {
        "name": "digit_without_evidence",
        "font_path": path,
        "missing_digits": list(font.missing_digits),
        "empty_digit_export": payload["digits"]["4"],
    }


# ---------------------------------------------------------------------------
# Scenario 6: multi-row font, leading zeros and swapping the batch contract
# ---------------------------------------------------------------------------


def scenario_multirow_and_contract_swap(outdir: str) -> dict:
    # Height-3 font with simple rectangular glyphs for 0, 1, 2.
    grids = {
        "0": [
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        "1": [
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        "2": [
            [1, 1],
            [0, 1],
            [1, 0],
        ],
    }
    runs = []
    sheets = []
    for digit, grid in grids.items():
        run = _run_from_grid(f"mr-run-{digit}", grid)
        runs.append(run)
        sheets.append(_sheet(f"mr-{digit}", digit, run, 3, 2, F(1), F(1)))
    font = build_font(runs, sheets, font_id="multirow-font")
    assert font.height == 3
    assert font.digit_widths == {"0": 2, "1": 2, "2": 2}
    path = os.path.join(outdir, "font_multirow.json")
    font.save(path)
    reader = LabelInterpreter.from_file(path)

    band = _band_from_digits(font, "012")  # 6 columns, leading zero is a character
    assert len(band) == 6
    # 0*4 + 1*2 + 2*1 = 4.  Modulus 7: remainder 4 -> unique "012".
    contract_ok = LabelContract.create((4, 2, 1), 7, [4], name="batch-A")
    answer = reader.solve(band, 3, contract_ok)
    assert answer.status == "unique" and answer.digits == ("012",)
    witness = answer.witnesses[0]
    assert [(c.column_start, c.column_end) for c in witness.characters] == [(0, 2), (2, 4), (4, 6)]
    assert witness.checksum == 4
    for char in witness.characters:
        # Whole-glyph source: 6 cell coordinates, one sheet, one threshold.
        assert len(char.origin.cells) == 6
        assert char.origin.threshold == F(1)

    # Re-read the SAME band with only the contract changed: now no solution.
    contract_other = LabelContract.create((4, 2, 1), 7, [0], name="batch-B")
    swapped = reader.solve(band, 3, contract_other)
    assert swapped.status == "no_solution"

    # A band with the wrong height or an uncovered column vector is rejected.
    try:
        reader.solve([[1, 0]], 1, LabelContract.create((1,), 7, [0]))
    except LabelReaderError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("wrong-height band must be rejected")
    unknown_vector = [[0, 0, 0], [0, 0, 0]]
    dead = reader.solve(unknown_vector, 1, LabelContract.create((1,), 7, [0]))
    assert dead.status == "no_solution"

    return {
        "name": "multirow_and_contract_swap",
        "font_path": path,
        "read_batch_a": answer.to_dict(),
        "read_batch_b_same_band": swapped.to_dict(),
    }


# ---------------------------------------------------------------------------
# Scenario 7: 80 digits / 480 columns, long identical-column runs
# ---------------------------------------------------------------------------


def scenario_large_band(outdir: str) -> dict:
    # Every digit gets one fixed six-column variant so 80 digits fill exactly
    # 480 columns.  All variants share the same repeated column vector, which
    # creates astronomically many equivalent cuts; the DP must stay exact and
    # bounded and never guess after an arbitrary attempt count.
    height = 2
    shared_column = (1, 0)
    runs = []
    sheets = []
    for digit in "0123456789":
        grid = [[1] * 6, [0] * 6]
        run = _run_from_grid(f"big-run-{digit}", grid)
        runs.append(run)
        sheets.append(_sheet(f"big-{digit}", digit, run, height, 6, F(1), F(1)))
    font = build_font(runs, sheets, font_id="big-font")
    path = os.path.join(outdir, "font_large.json")
    font.save(path)
    reader = LabelInterpreter.from_file(path)

    band = [list(shared_column) for _ in range(480)]
    modulus = 97
    # Weights all 1: checksum = sum of digits.  Two disjoint remainder sets
    # give one unique answer vs an ambiguous pair on the same band.
    weights = tuple([1] * 80)
    # 80 zeros is always available and has checksum 0.
    contract_zero = LabelContract.create(weights, modulus, [0], name="all-zero-only")
    zero_answer = reader.solve(band, 80, contract_zero)
    assert zero_answer.status == "unique"
    assert zero_answer.digits == ("0" * 80,)
    # Allowing remainder 1 as well admits exactly the strings with one digit 1
    # (80 such strings); we must report two distinct ones, not guess.
    contract_two = LabelContract.create(weights, modulus, [0, 1], name="zero-or-one")
    ambiguous = reader.solve(band, 80, contract_two)
    assert ambiguous.status == "ambiguous"
    first, second = ambiguous.witnesses
    assert first.digits != second.digits
    assert {sum(int(d) for d in w.digits) for w in ambiguous.witnesses} == {0, 1}
    # Both witnesses cover the full 480 columns in 80 slices of width 6.
    for witness in ambiguous.witnesses:
        intervals = [(c.column_start, c.column_end) for c in witness.characters]
        assert intervals[0] == (0, 6)
        assert intervals[-1] == (474, 480)
        assert all(end - start == 6 for start, end in intervals)

    return {
        "name": "large_band_80x480",
        "font_path": path,
        "unique_digest": zero_answer.digits[0][:12] + "...",
        "ambiguous_pair": [w.digits[:12] + "..." for w in ambiguous.witnesses],
        "status": ambiguous.status,
    }


def build_all_demos(outdir: str = "results") -> dict:
    os.makedirs(outdir, exist_ok=True)
    demos = {
        "canonical_one_two": scenario_canonical_one_two,
        "duplicate_sheets": scenario_duplicate_sheets,
        "threshold_representatives": scenario_threshold_representatives,
        "missing_cells": scenario_missing_cells,
        "digit_without_evidence": scenario_digit_without_evidence,
        "multirow_and_contract_swap": scenario_multirow_and_contract_swap,
        "large_band": scenario_large_band,
    }
    results = {}
    for name, builder in demos.items():
        results[name] = builder(outdir)
        _dump(os.path.join(outdir, f"label_{name}.json"), results[name])
    return results


def main(argv: list[str] | None = None) -> int:
    import sys
    import time

    argv = list(sys.argv[1:] if argv is None else argv)
    outdir = argv[0] if argv else "results"
    started = time.perf_counter()
    results = build_all_demos(outdir)
    elapsed = time.perf_counter() - started
    for name, payload in results.items():
        print(f"{name}: OK")
    print(f"all demos wrote fonts and reports under {outdir}/ in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
