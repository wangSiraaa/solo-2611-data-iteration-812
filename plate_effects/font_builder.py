"""Digital glyph font built from whole-exposure synthetic correction runs.

Model
-----
A correction run owns corrected well values.  A *sample sheet* maps one
rectangular hole array (``height`` rows x ``width`` columns, the cell grid of
one glyph) onto wells of one run and records the *closed rational threshold
interval* [``t_lo``, ``t_hi``] of the single exposure that produced it.

The exposure shares one unknown threshold across every cell of the sheet.
A cell is lit exactly when its corrected value is greater than or equal to
that threshold::

    cell(row, col) lit  <=>  corrected_value(row, col) >= t

so a legal glyph bitmap must come from *one complete sheet at one threshold*.
Columns from two sheets are never spliced: each exported variant keeps the
sheet, the representative threshold and the source well coordinates of its
whole bitmap.

As ``t`` moves inside the closed interval the bitmap only changes when ``t``
crosses a cell value.  All distinct bitmaps are therefore enumerated exactly
by evaluating the interval endpoints and every in-range cell value, over
``fractions.Fraction`` with no tolerance.  Missing cells are never painted
dark: a sheet with any unmeasured (or unrecognisable) cell yields no bitmap at
all and is exported as an invalid-sheet record instead.

Bounds: height <= 8, width <= 12, digits 0..9.  Every digit of a font has one
fixed width; all digits share the font height.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .rational import RationalError, fmt_rational, parse_rational

FONT_FORMAT = "glyph-font/v1"
MAX_GLYPH_HEIGHT = 8
MAX_GLYPH_WIDTH = 12
DIGITS = tuple("0123456789")

# Invalid-sheet reason codes.
INVALID_MISSING_CELL = "MISSING_CELL"
INVALID_UNRECOGNISABLE_CELL = "UNRECOGNISABLE_CELL"
INVALID_EMPTY_INTERVAL = "EMPTY_THRESHOLD_INTERVAL"
INVALID_DUPLICATE_MAPPING = "DUPLICATE_WELL_MAPPING"
INVALID_OUT_OF_RUN = "WELL_NOT_IN_RUN"


class FontError(ValueError):
    """Raised when font input violates the glyph contract."""


# ---------------------------------------------------------------------------
# Input records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateHole:
    """One corrected hole of a run: well name, corrected value and its status.

    ``corrected`` is ``None`` for holes the run could not correct
    (``MISSING`` / ``UNIDENTIFIABLE``); such holes cannot anchor a glyph cell.
    """

    run_id: str
    well: str
    corrected: Fraction | None
    status: str

    @classmethod
    def from_well_result(cls, run_id: str, result: object) -> "PlateHole":
        return cls(
            run_id=run_id,
            well=result.well,
            corrected=result.corrected,
            status=result.status,
        )


@dataclass(frozen=True)
class CorrectionRun:
    """The subset of a warehouse run needed by the font builder."""

    run_id: str
    holes: Mapping[str, PlateHole]

    @classmethod
    def from_warehouse_run(cls, run: object, report: object) -> "CorrectionRun":
        """Build from a ``ReadoutRun`` and its fitted ``FitReport``."""

        holes = {
            result.well: PlateHole.from_well_result(run.run_id, result)
            for result in report.wells
        }
        return cls(run_id=run.run_id, holes=holes)

    @classmethod
    def from_values(cls, run_id: str, values: Mapping[str, object]) -> "CorrectionRun":
        """Build from a plain ``well -> rational`` mapping (synthetic runs).

        ``None`` values encode an unmeasured/uncorrectable hole.
        """

        holes: dict[str, PlateHole] = {}
        for well, raw in values.items():
            if raw is None:
                holes[well] = PlateHole(run_id, well, None, "MISSING")
            else:
                holes[well] = PlateHole(
                    run_id, well, parse_rational(raw, field=f"hole {well}"), "CORRECTED"
                )
        return cls(run_id=run_id, holes=holes)


@dataclass(frozen=True)
class SampleSheet:
    """One exposure: a rectangular cell grid mapped onto one run's wells.

    ``cells`` maps ``(row, col)`` (zero-based inside the glyph rectangle) to a
    well name inside ``run_id``.  The threshold interval is closed: both
    endpoints are legal exposure thresholds.
    """

    sheet_id: str
    digit: str
    run_id: str
    height: int
    width: int
    cells: Mapping[tuple[int, int], str]
    t_lo: Fraction
    t_hi: Fraction

    def __post_init__(self) -> None:
        if self.digit not in DIGITS:
            raise FontError(f"digit must be one of {''.join(DIGITS)}, got {self.digit!r}")
        if not 1 <= self.height <= MAX_GLYPH_HEIGHT:
            raise FontError(f"height must be in 1..{MAX_GLYPH_HEIGHT}")
        if not 1 <= self.width <= MAX_GLYPH_WIDTH:
            raise FontError(f"width must be in 1..{MAX_GLYPH_WIDTH}")
        if not isinstance(self.t_lo, Fraction) or not isinstance(self.t_hi, Fraction):
            raise FontError("threshold endpoints must be exact rationals")
        if self.t_lo > self.t_hi:
            raise FontError("threshold interval is empty: t_lo > t_hi")
        expected = {(r, c) for r in range(self.height) for c in range(self.width)}
        if set(self.cells) != expected:
            raise FontError(
                f"sheet {self.sheet_id!r} must map exactly the {self.height}x{self.width} "
                f"rectangle, once each"
            )
        if len(set(self.cells.values())) != len(self.cells):
            raise FontError(f"sheet {self.sheet_id!r} maps two cells to the same well")


# ---------------------------------------------------------------------------
# Exported font records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSource:
    """Source coordinates of one glyph cell inside a sample sheet."""

    row: int
    col: int
    well: str
    corrected: Fraction

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "well": self.well,
            "corrected": fmt_rational(self.corrected),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CellSource":
        return cls(
            row=int(payload["row"]),
            col=int(payload["col"]),
            well=str(payload["well"]),
            corrected=parse_rational(payload["corrected"], field="cell.corrected"),
        )


@dataclass(frozen=True)
class GlyphOrigin:
    """Verifiable provenance of one whole bitmap: one sheet, one threshold.

    The bitmap is reconstructible by re-applying the shared threshold to the
    per-cell corrected values; it can never be a column-by-column patchwork.
    """

    sheet_id: str
    run_id: str
    digit: str
    threshold: Fraction
    threshold_interval: tuple[Fraction, Fraction]
    cells: tuple[CellSource, ...]

    def to_dict(self) -> dict:
        return {
            "sheet_id": self.sheet_id,
            "run_id": self.run_id,
            "digit": self.digit,
            "threshold": fmt_rational(self.threshold),
            "threshold_interval": [
                fmt_rational(self.threshold_interval[0]),
                fmt_rational(self.threshold_interval[1]),
            ],
            "cells": [cell.to_dict() for cell in sorted(self.cells, key=lambda c: (c.row, c.col))],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "GlyphOrigin":
        interval = payload["threshold_interval"]
        return cls(
            sheet_id=payload["sheet_id"],
            run_id=payload["run_id"],
            digit=payload["digit"],
            threshold=parse_rational(payload["threshold"], field="origin.threshold"),
            threshold_interval=(
                parse_rational(interval[0], field="t_lo"),
                parse_rational(interval[1], field="t_hi"),
            ),
            cells=tuple(CellSource.from_dict(item) for item in payload["cells"]),
        )


@dataclass(frozen=True)
class GlyphVariant:
    """One distinct column bitmap of one digit, with every witness origin."""

    variant_id: str
    digit: str
    width: int
    height: int
    # columns[col] is a tuple of 0/1 rows, row 0 first.
    columns: tuple[tuple[int, ...], ...]
    origins: tuple[GlyphOrigin, ...]

    @property
    def bitmap(self) -> tuple[tuple[int, ...], ...]:
        """Rows of 0/1, row 0 first."""

        return tuple(
            tuple(self.columns[c][r] for c in range(self.width)) for r in range(self.height)
        )

    def render(self) -> str:
        return "\n".join("".join("#" if bit else "." for bit in row) for row in self.bitmap)

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "digit": self.digit,
            "width": self.width,
            "height": self.height,
            "columns": [list(column) for column in self.columns],
            "bitmap": [list(row) for row in self.bitmap],
            "origins": [origin.to_dict() for origin in self.origins],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "GlyphVariant":
        columns = tuple(tuple(int(bit) for bit in column) for column in payload["columns"])
        return cls(
            variant_id=payload["variant_id"],
            digit=payload["digit"],
            width=int(payload["width"]),
            height=int(payload["height"]),
            columns=columns,
            origins=tuple(GlyphOrigin.from_dict(item) for item in payload["origins"]),
        )


@dataclass(frozen=True)
class InvalidSheet:
    """A sheet that cannot produce a legal bitmap; missing cells stay missing."""

    sheet_id: str
    digit: str
    run_id: str
    reason: str
    detail: str
    cells: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict:
        return {
            "sheet_id": self.sheet_id,
            "digit": self.digit,
            "run_id": self.run_id,
            "reason": self.reason,
            "detail": self.detail,
            "cells": [list(cell) for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "InvalidSheet":
        return cls(
            sheet_id=payload["sheet_id"],
            digit=payload["digit"],
            run_id=payload["run_id"],
            reason=payload["reason"],
            detail=payload["detail"],
            cells=tuple((int(r), int(c)) for r, c in payload["cells"]),
        )


@dataclass(frozen=True)
class GlyphFont:
    """Exportable font product: fixed-height grid, one fixed width per digit."""

    font_id: str
    height: int
    digit_widths: Mapping[str, int]
    variants: Mapping[str, tuple[GlyphVariant, ...]]
    invalid_sheets: tuple[InvalidSheet, ...]
    missing_digits: tuple[str, ...]
    digest: str
    sheet_count: int

    # -- views ----------------------------------------------------------------
    def digits(self) -> tuple[str, ...]:
        return tuple(d for d in DIGITS if d in self.digit_widths)

    def variant(self, variant_id: str) -> GlyphVariant:
        for variants in self.variants.values():
            for candidate in variants:
                if candidate.variant_id == variant_id:
                    return candidate
        raise KeyError(variant_id)

    def to_dict(self) -> dict:
        return {
            "format": FONT_FORMAT,
            "font_id": self.font_id,
            "height": self.height,
            "digit_widths": dict(self.digit_widths),
            "missing_digits": list(self.missing_digits),
            "digits": {
                digit: [variant.to_dict() for variant in self.variants.get(digit, ())]
                for digit in DIGITS
            },
            "invalid_sheets": [sheet.to_dict() for sheet in self.invalid_sheets],
            "sheet_count": self.sheet_count,
            "digest": self.digest,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "GlyphFont":
        if payload.get("format") != FONT_FORMAT:
            raise FontError(f"unsupported font format {payload.get('format')!r}")
        height = int(payload["height"])
        digit_widths = {str(d): int(w) for d, w in payload["digit_widths"].items()}
        variants: dict[str, tuple[GlyphVariant, ...]] = {}
        for digit, records in payload["digits"].items():
            if records:
                variants[digit] = tuple(GlyphVariant.from_dict(item) for item in records)
        invalid = tuple(InvalidSheet.from_dict(item) for item in payload["invalid_sheets"])
        font = cls(
            font_id=payload["font_id"],
            height=height,
            digit_widths=digit_widths,
            variants=variants,
            invalid_sheets=invalid,
            missing_digits=tuple(payload["missing_digits"]),
            digest=payload["digest"],
            sheet_count=int(payload["sheet_count"]),
        )
        if font_digest(font._semantics()) != font.digest:
            raise FontError("font digest does not match its content")
        return font

    @classmethod
    def load(cls, path: str) -> "GlyphFont":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def _semantics(self) -> dict:
        """Content covered by the digest (excludes the digest field itself)."""

        return {
            "format": FONT_FORMAT,
            "font_id": self.font_id,
            "height": self.height,
            "digit_widths": {d: self.digit_widths[d] for d in sorted(self.digit_widths)},
            "variants": [
                variant.to_dict()
                for digit in DIGITS
                for variant in self.variants.get(digit, ())
            ],
            "invalid_sheets": [sheet.to_dict() for sheet in self.invalid_sheets],
        }


def font_digest(semantics: Mapping) -> str:
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Whole-exposure bitmap enumeration
# ---------------------------------------------------------------------------


def _threshold_representatives(
    cell_values: Sequence[Fraction], t_lo: Fraction, t_hi: Fraction
) -> tuple[Fraction, ...]:
    """One threshold per distinct bitmap the closed interval can produce.

    A bitmap changes only when the shared threshold crosses a cell value.
    The closed endpoints are themselves legal thresholds, and every in-range
    cell value is a crossing point.  Evaluating exactly those rationals covers
    every region of constant bitmap; duplicates are removed.
    """

    candidates = {t_lo, t_hi}
    for value in cell_values:
        if t_lo <= value <= t_hi:
            candidates.add(value)
    return tuple(sorted(candidates))


def _bitmap_at(
    cell_values: Mapping[tuple[int, int], Fraction],
    height: int,
    width: int,
    threshold: Fraction,
) -> tuple[tuple[int, ...], ...]:
    """Columns of 0/1 under one shared threshold (lit iff value >= t)."""

    return tuple(
        tuple(
            1 if cell_values[(r, c)] >= threshold else 0 for r in range(height)
        )
        for c in range(width)
    )


def build_font(
    runs: Sequence[CorrectionRun],
    sheets: Sequence[SampleSheet],
    *,
    font_id: str = "font-0001",
) -> GlyphFont:
    """Build an exportable glyph font from corrected runs and sample sheets.

    A valid sheet contributes every distinct bitmap its closed threshold
    interval can produce.  Invalid sheets are recorded with a reason and
    contribute nothing; missing cells are never silently painted dark.
    """

    run_map = {run.run_id: run for run in runs}
    if len(run_map) != len(runs):
        raise FontError("duplicate run id")

    # Per-digit merged bitmap -> origins, and per-digit fixed width.
    merged: dict[str, dict[tuple[tuple[int, ...], ...], list[GlyphOrigin]]] = {}
    digit_widths: dict[str, int] = {}
    invalid: list[InvalidSheet] = []

    for sheet in sheets:
        run = run_map.get(sheet.run_id)
        if run is None:
            raise FontError(f"sheet {sheet.sheet_id!r} references unknown run {sheet.run_id!r}")

        missing: list[tuple[int, int]] = []
        unrecognisable: list[tuple[int, int]] = []
        out_of_run: list[tuple[int, int]] = []
        values: dict[tuple[int, int], Fraction] = {}
        for (r, c), well in sheet.cells.items():
            hole = run.holes.get(well)
            if hole is None:
                out_of_run.append((r, c))
            elif hole.corrected is None:
                if hole.status == "MISSING":
                    missing.append((r, c))
                else:
                    unrecognisable.append((r, c))
            else:
                values[(r, c)] = hole.corrected

        if out_of_run:
            invalid.append(
                InvalidSheet(
                    sheet.sheet_id,
                    sheet.digit,
                    sheet.run_id,
                    INVALID_OUT_OF_RUN,
                    "mapped well is absent from the run",
                    tuple(sorted(out_of_run)),
                )
            )
            continue
        if missing:
            invalid.append(
                InvalidSheet(
                    sheet.sheet_id,
                    sheet.digit,
                    sheet.run_id,
                    INVALID_MISSING_CELL,
                    "at least one cell has no corrected value; missing cells are not painted dark",
                    tuple(sorted(missing)),
                )
            )
            continue
        if unrecognisable:
            invalid.append(
                InvalidSheet(
                    sheet.sheet_id,
                    sheet.digit,
                    sheet.run_id,
                    INVALID_UNRECOGNISABLE_CELL,
                    "at least one cell is unidentifiable and cannot be thresholded",
                    tuple(sorted(unrecognisable)),
                )
            )
            continue

        previous_width = digit_widths.get(sheet.digit)
        if previous_width is not None and previous_width != sheet.width:
            raise FontError(
                f"digit {sheet.digit} has conflicting widths {previous_width} and {sheet.width}"
            )
        digit_widths[sheet.digit] = sheet.width

        seen_bitmaps: dict[tuple[tuple[int, ...], ...], Fraction] = {}
        for threshold in _threshold_representatives(
            list(values.values()), sheet.t_lo, sheet.t_hi
        ):
            columns = _bitmap_at(values, sheet.height, sheet.width, threshold)
            if columns in seen_bitmaps:
                # Same bitmap at a second threshold: the first origin already
                # witnesses it; keep one origin per (sheet, bitmap).
                continue
            seen_bitmaps[columns] = threshold
            cells = tuple(
                CellSource(row=r, col=c, well=sheet.cells[(r, c)], corrected=values[(r, c)])
                for r in range(sheet.height)
                for c in range(sheet.width)
            )
            origin = GlyphOrigin(
                sheet_id=sheet.sheet_id,
                run_id=sheet.run_id,
                digit=sheet.digit,
                threshold=threshold,
                threshold_interval=(sheet.t_lo, sheet.t_hi),
                cells=cells,
            )
            merged.setdefault(sheet.digit, {}).setdefault(columns, []).append(origin)

    height = _common_height(sheets)
    variants: dict[str, tuple[GlyphVariant, ...]] = {}
    for digit in DIGITS:
        bitmap_map = merged.get(digit)
        if not bitmap_map:
            continue
        digit_variants = []
        for index, (columns, origins) in enumerate(
            sorted(bitmap_map.items(), key=lambda item: item[0]), start=1
        ):
            digit_variants.append(
                GlyphVariant(
                    variant_id=f"{digit}-v{index:02d}",
                    digit=digit,
                    width=digit_widths[digit],
                    height=height,
                    columns=columns,
                    origins=tuple(origins),
                )
            )
        variants[digit] = tuple(digit_variants)

    missing_digits = tuple(d for d in DIGITS if d not in digit_widths)
    font = GlyphFont(
        font_id=font_id,
        height=height,
        digit_widths=digit_widths,
        variants=variants,
        invalid_sheets=tuple(invalid),
        missing_digits=missing_digits,
        digest="",
        sheet_count=len(sheets),
    )
    digest = font_digest(font._semantics())
    return GlyphFont(
        font_id=font.font_id,
        height=font.height,
        digit_widths=dict(font.digit_widths),
        variants=dict(font.variants),
        invalid_sheets=font.invalid_sheets,
        missing_digits=font.missing_digits,
        digest=digest,
        sheet_count=font.sheet_count,
    )


def _common_height(sheets: Sequence[SampleSheet]) -> int:
    heights = {sheet.height for sheet in sheets}
    if not heights:
        raise FontError("at least one sample sheet is required")
    if len(heights) > 1:
        raise FontError(f"all glyphs must share one height, got {sorted(heights)}")
    return heights.pop()


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def sheet_from_rectangle(
    sheet_id: str,
    digit: str,
    run: CorrectionRun,
    *,
    height: int,
    width: int,
    wells: Sequence[str],
    t_lo: object,
    t_hi: object,
) -> SampleSheet:
    """Build a sheet from a row-major well list (row 0 first, then columns)."""

    if len(wells) != height * width:
        raise FontError(
            f"sheet {sheet_id!r}: expected {height * width} wells, got {len(wells)}"
        )
    cells = {
        (r, c): wells[r * width + c]
        for r in range(height)
        for c in range(width)
    }
    try:
        lo = parse_rational(t_lo, field="t_lo")
        hi = parse_rational(t_hi, field="t_hi")
    except RationalError as exc:
        raise FontError(str(exc)) from exc
    return SampleSheet(
        sheet_id=sheet_id,
        digit=digit,
        run_id=run.run_id,
        height=height,
        width=width,
        cells=cells,
        t_lo=lo,
        t_hi=hi,
    )
