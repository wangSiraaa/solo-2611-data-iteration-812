"""Batch readout warehouse with exact additive row/column bias correction.

Model
-----
For every *used* well ``(i, j)`` of a plate::

    reference response = expected + b + r_i + c_j + error
    blank     response =          0 + b + r_i + c_j + error
    sample    response =       value + b + r_i + c_j + error

``b`` is the plate offset, ``r`` the row effect, ``c`` the column effect.
The identifiability constraint is::

    sum(r_i over existing rows) = 0
    sum(c_j over existing columns) = 0

"Existing" means the rows and columns of the plate geometry (the whole
configured grid), so the parameter count is ``1 + (rows-1) + (cols-1)`` and is
a layout-level fact cached per digest (see ``_structure_for``).

Fitting is done over the rationals with exact Gaussian elimination on the
normal equations of the observed control wells.  Identifiability is decided
from the *exact* nullspace of the constrained control design matrix: a sample
well bias ``b + r_i + c_j`` is unique across all least-squares minimisers iff
its coefficient vector is orthogonal to that nullspace.  No tolerance is used
for any decision; a numerical rank is reported only as a cross-check.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np

from .design import (
    ROLE_BLANK,
    ROLE_REFERENCE,
    ROLE_UNUSED,
    PublishedLayout,
    PublishedLayoutError,
    Well,
    WellError,
    _layout_semantics,
    layout_digest,
)
from .rational import (
    Matrix,
    RationalError,
    Vector,
    dot,
    fmt_rational,
    matvec,
    parse_rational,
)

MAX_PLATES = 30
NUMERIC_RANK_TOLERANCE = 1e-9

STATUS_CORRECTED = "CORRECTED"
STATUS_MISSING = "MISSING"
STATUS_UNIDENTIFIABLE = "UNIDENTIFIABLE"
STATUS_CONTROL = "CONTROL"


class WarehouseError(ValueError):
    """Base class for readout-warehouse errors."""


class ImportError_(WarehouseError):
    """Raised when a readout import violates the import contract."""


class RunNotFound(WarehouseError):
    pass


# ---------------------------------------------------------------------------
# Layout structure shared by all runs of a layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Structure:
    """Layout-level facts shared by every run of one layout digest.

    "Existing" rows and columns are the rows and columns of the plate itself
    (the configured geometry), not merely the ones that happen to carry a used
    well.  A plate of at most 8x12 therefore always has exactly
    ``rows + cols - 1`` bias parameters: ``b``, the row effects with one
    zero-sum constraint, and the column effects with one zero-sum constraint.

    Parameter coordinates: index 0 is ``b``; the next ``len(rows)-1`` are the
    effects of all but the last row; the last row is fixed by ``sum(r) = 0`` to
    the negative of their sum.  Columns follow the same pattern.  A single
    existing row/column therefore has effect exactly zero.
    """

    digest: str
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    n_params: int

    @property
    def row_coords(self) -> int:
        return max(len(self.rows) - 1, 0)

    @property
    def col_coords(self) -> int:
        return max(len(self.cols) - 1, 0)


_STRUCTURE_CACHE: "OrderedDict[str, _Structure]" = OrderedDict()
_STRUCTURE_CACHE_LIMIT = 512


def _structure_for(layout: PublishedLayout) -> _Structure:
    cached = _STRUCTURE_CACHE.get(layout.digest)
    if cached is not None:
        _STRUCTURE_CACHE.move_to_end(layout.digest)
        return cached

    rows = tuple(range(layout.rows))
    cols = tuple(range(layout.cols))
    n_params = 1 + (layout.rows - 1) + (layout.cols - 1)

    structure = _Structure(
        digest=layout.digest,
        rows=rows,
        cols=cols,
        n_params=n_params,
    )
    _STRUCTURE_CACHE[layout.digest] = structure
    _STRUCTURE_CACHE.move_to_end(layout.digest)
    while len(_STRUCTURE_CACHE) > _STRUCTURE_CACHE_LIMIT:
        _STRUCTURE_CACHE.popitem(last=False)
    return structure


# Parameter-space coordinate layout is described on ``_Structure``.  These two
# helpers turn coordinates into a well's bias feature and back into effects.
def _feature(structure: _Structure, row: int, col: int) -> Vector:
    vec = [Fraction(0) for _ in range(structure.n_params)]
    if structure.n_params == 0:
        return vec
    vec[0] = Fraction(1)
    row_coords = structure.row_coords
    if row_coords:
        if row == structure.rows[-1]:
            for index in range(row_coords):
                vec[1 + index] -= 1
        else:
            vec[1 + structure.rows.index(row)] += 1
    col_base = 1 + row_coords
    col_coords = structure.col_coords
    if col_coords:
        if col == structure.cols[-1]:
            for index in range(col_coords):
                vec[col_base + index] -= 1
        else:
            vec[col_base + structure.cols.index(col)] += 1
    return vec


def _effect_from_params(structure: _Structure, vec: Sequence[Fraction]) -> dict[str, object]:
    if structure.n_params == 0:
        return {"b": Fraction(0), "rows": {}, "cols": {}}
    b = vec[0]
    row_coords = structure.row_coords
    rows: dict[int, Fraction] = {}
    if row_coords == 0:
        rows[structure.rows[0]] = Fraction(0)
    else:
        total = Fraction(0)
        for index, row in enumerate(structure.rows[:-1]):
            value = vec[1 + index]
            rows[row] = value
            total += value
        rows[structure.rows[-1]] = -total
    col_base = 1 + row_coords
    col_coords = structure.col_coords
    cols: dict[int, Fraction] = {}
    if col_coords == 0:
        cols[structure.cols[0]] = Fraction(0)
    else:
        total = Fraction(0)
        for index, col in enumerate(structure.cols[:-1]):
            value = vec[col_base + index]
            cols[col] = value
            total += value
        cols[structure.cols[-1]] = -total
    return {"b": b, "rows": rows, "cols": cols}


# ---------------------------------------------------------------------------
# Cached exact factorisation of the control system
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ControlFactor:
    """Everything that depends only on (layout digest, observed control set)."""

    digest: str
    control_wells: tuple[str, ...]
    n_params: int
    design: Matrix
    normal: Matrix
    rref_normal: Matrix
    transform: Matrix
    pivots: tuple[int, ...]
    null_basis: tuple[Vector, ...]
    rank: int
    numerical_rank: int
    rank_tolerance: float

    @property
    def n_equations(self) -> int:
        return len(self.design)

    @property
    def nullity(self) -> int:
        return self.n_params - self.rank


_FACTOR_CACHE: "OrderedDict[tuple[str, tuple[str, ...]], _ControlFactor]" = OrderedDict()
_FACTOR_CACHE_LIMIT = 256


def _rref_with_transform(matrix: Sequence[Sequence[Fraction]]) -> tuple[Matrix, Matrix, list[int]]:
    """RREF plus the row-operation matrix ``T`` with ``T @ matrix == rref``."""

    p = len(matrix)
    if p == 0:
        return [], [], []
    width = len(matrix[0])
    work = [list(row) + [Fraction(1 if i == j else 0) for j in range(p)] for i, row in enumerate(matrix)]
    pivots: list[int] = []
    r = 0
    for c in range(width):
        pivot = next((i for i in range(r, p) if work[i][c] != 0), None)
        if pivot is None:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        pv = work[r][c]
        if pv != 1:
            work[r] = [x / pv for x in work[r]]
        for i in range(p):
            if i == r:
                continue
            factor = work[i][c]
            if factor:
                work[i] = [x - factor * y for x, y in zip(work[i], work[r])]
        pivots.append(c)
        r += 1
        if r == p:
            break
    return [row[:width] for row in work], [row[width:] for row in work], pivots


def _numerical_rank(design: Sequence[Sequence[Fraction]]) -> float:
    if not design:
        return 0, NUMERIC_RANK_TOLERANCE
    matrix = np.array([[float(x) for x in row] for row in design], dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0, NUMERIC_RANK_TOLERANCE
    sigma_max = float(singular[0])
    tol = NUMERIC_RANK_TOLERANCE * max(1.0, sigma_max)
    return int(np.sum(singular > tol)), tol


def _build_factor(
    layout: PublishedLayout,
    structure: _Structure,
    control_wells: tuple[str, ...],
) -> _ControlFactor:
    well_map = layout.well_map
    design: Matrix = []
    for name in control_wells:
        well = well_map[name]
        design.append(_feature(structure, well.row, well.col))
    p = structure.n_params
    m = len(design)
    normal = [[Fraction(0) for _ in range(p)] for _ in range(p)]
    for row in design:
        for i in range(p):
            if row[i] == 0:
                continue
            for j in range(p):
                if row[j]:
                    normal[i][j] += row[i] * row[j]
    rref_normal, transform, pivots = _rref_with_transform(normal)
    rank = len(pivots)
    pivot_set = set(pivots)
    null_basis: list[Vector] = []
    for free_col in range(p):
        if free_col in pivot_set:
            continue
        vec = [Fraction(0) for _ in range(p)]
        vec[free_col] = Fraction(1)
        for row_index, pivot_col in enumerate(pivots):
            vec[pivot_col] = -rref_normal[row_index][free_col]
        null_basis.append(vec)
    numerical_rank, tol = _numerical_rank(design)
    return _ControlFactor(
        digest=layout.digest,
        control_wells=control_wells,
        n_params=p,
        design=design,
        normal=normal,
        rref_normal=rref_normal,
        transform=transform,
        pivots=tuple(pivots),
        null_basis=tuple(null_basis),
        rank=rank,
        numerical_rank=numerical_rank,
        rank_tolerance=tol,
    )


def _factor_for(layout: PublishedLayout, control_wells: Sequence[str]) -> _ControlFactor:
    key = (layout.digest, tuple(control_wells))
    cached = _FACTOR_CACHE.get(key)
    if cached is not None:
        _FACTOR_CACHE.move_to_end(key)
        return cached
    structure = _structure_for(layout)
    factor = _build_factor(layout, structure, tuple(control_wells))
    _FACTOR_CACHE[key] = factor
    _FACTOR_CACHE.move_to_end(key)
    while len(_FACTOR_CACHE) > _FACTOR_CACHE_LIMIT:
        _FACTOR_CACHE.popitem(last=False)
    return factor


def clear_caches() -> None:
    _STRUCTURE_CACHE.clear()
    _FACTOR_CACHE.clear()


# ---------------------------------------------------------------------------
# Runs and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutRun:
    """A batch readout bound to an immutable snapshot of one layout version."""

    run_id: str
    layout: PublishedLayout
    measurements: Mapping[str, Fraction]
    note: str | None = None

    @property
    def plate_id(self) -> str:
        return self.layout.plate_id

    @property
    def layout_version(self) -> int:
        return self.layout.version

    @property
    def layout_digest(self) -> str:
        return self.layout.digest

    @property
    def missing_used_wells(self) -> tuple[str, ...]:
        used = {w.name for w in self.layout.wells if w.role != ROLE_UNUSED}
        return tuple(sorted(used - set(self.measurements)))


@dataclass(frozen=True)
class WellResult:
    well: str
    row: int
    col: int
    role: str
    sample_id: str | None
    expected: Fraction | None
    observed: Fraction | None
    bias: Fraction | None
    bias_identifiable: bool
    corrected: Fraction | None
    residual: Fraction | None
    status: str
    proof: dict | None = None
    proof_direction: Vector | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        def rat(value: Fraction | None) -> dict | None:
            if value is None:
                return None
            return {"exact": fmt_rational(value), "float": float(value)}

        return {
            "well": self.well,
            "row": self.row,
            "col": self.col,
            "role": self.role,
            "sample_id": self.sample_id,
            "expected": rat(self.expected),
            "observed": rat(self.observed),
            "bias": rat(self.bias),
            "bias_identifiable": self.bias_identifiable,
            "corrected": rat(self.corrected),
            "residual": rat(self.residual),
            "status": self.status,
            "unidentifiability_proof": self.proof,
        }


@dataclass(frozen=True)
class SampleSummary:
    sample_id: str
    wells: tuple[str, ...]
    status: str
    corrected_values: dict[str, Fraction]
    mean: Fraction | None
    range: Fraction | None
    reasons: tuple[dict, ...]

    def to_dict(self) -> dict:
        def rat(value: Fraction | None) -> dict | None:
            if value is None:
                return None
            return {"exact": fmt_rational(value), "float": float(value)}

        return {
            "sample_id": self.sample_id,
            "wells": list(self.wells),
            "status": self.status,
            "corrected_values": {
                name: {"exact": fmt_rational(value), "float": float(value)}
                for name, value in self.corrected_values.items()
            },
            "mean": rat(self.mean),
            "range": rat(self.range),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FitReport:
    run_id: str
    plate_id: str
    layout_version: int
    layout_digest: str
    control_wells: tuple[str, ...]
    missing_control_wells: tuple[str, ...]
    observed_control_count: int
    rank: int
    n_params: int
    nullity: int
    numerical_rank: int
    rank_tolerance: float
    residual_sum_of_squares: Fraction
    max_abs_control_residual: Fraction
    parameter_unique: bool
    effects: dict[str, object]
    wells: tuple[WellResult, ...]
    samples: tuple[SampleSummary, ...]
    null_basis: tuple[Vector, ...]
    verification: dict[str, object]

    # -- views -----------------------------------------------------------------
    def well(self, value: object) -> WellResult:
        name = value if isinstance(value, str) else Well.parse(value, MAX_ROWS, MAX_COLS).name
        for result in self.wells:
            if result.well == name:
                return result
        raise KeyError(name)

    def sample(self, sample_id: str) -> SampleSummary:
        for summary in self.samples:
            if summary.sample_id == sample_id:
                return summary
        raise KeyError(sample_id)

    @property
    def unidentifiable_wells(self) -> tuple[str, ...]:
        return tuple(w.well for w in self.wells if w.status == STATUS_UNIDENTIFIABLE)

    @property
    def missing_wells(self) -> tuple[str, ...]:
        return tuple(w.well for w in self.wells if w.status == STATUS_MISSING)

    def to_dict(self) -> dict:
        def rat(value: Fraction | None) -> dict | None:
            if value is None:
                return None
            return {"exact": fmt_rational(value), "float": float(value)}

        effects = self.effects
        return {
            "run_id": self.run_id,
            "plate_id": self.plate_id,
            "layout_version": self.layout_version,
            "layout_digest": self.layout_digest,
            "control_wells": list(self.control_wells),
            "missing_control_wells": list(self.missing_control_wells),
            "observed_control_count": self.observed_control_count,
            "linear_system": {
                "rank": self.rank,
                "n_parameters": self.n_params,
                "nullity": self.nullity,
                "numerical_rank_crosscheck": self.numerical_rank,
                "numerical_rank_tolerance": self.rank_tolerance,
            },
            "control_residuals": {
                "sum_of_squares": rat(self.residual_sum_of_squares),
                "max_abs": rat(self.max_abs_control_residual),
                "per_well": {
                    w.well: rat(w.residual)
                    for w in self.wells
                    if w.role in (ROLE_BLANK, ROLE_REFERENCE) and w.observed is not None
                },
            },
            "parameter_unique": self.parameter_unique,
            "one_optimal_solution": {
                "b": rat(effects["b"]),
                "rows": {str(k): rat(v) for k, v in effects["rows"].items()},
                "cols": {str(k): rat(v) for k, v in effects["cols"].items()},
            },
            "wells": [w.to_dict() for w in self.wells],
            "samples": [s.to_dict() for s in self.samples],
            "verification": self.verification,
        }


# ---------------------------------------------------------------------------
# Warehouse
# ---------------------------------------------------------------------------


def _coerce_layout(layout: PublishedLayout | Mapping) -> PublishedLayout:
    if isinstance(layout, PublishedLayout):
        semantics = _layout_semantics(
            layout.plate_id,
            layout.rows,
            layout.cols,
            layout.version,
            layout.min_replicates,
            layout.max_replicates,
            list(layout.sample_bounds),
            layout.wells,
        )
        if layout_digest(semantics) != layout.digest:
            raise ImportError_("layout object digest does not match its content")
        return layout
    if isinstance(layout, Mapping):
        try:
            return PublishedLayout.from_dict(layout)
        except PublishedLayoutError as exc:
            raise ImportError_(f"external layout product rejected: {exc}") from exc
    raise ImportError_(f"unsupported layout type {type(layout).__name__}")


def _normalise_measurements(
    layout: PublishedLayout, measurements: object
) -> dict[str, Fraction]:
    """Normalise measurement input, rejecting every ambiguous form.

    Accepts a mapping ``well -> value`` or a sequence of records
    ``{"well": ..., "value": ...}`` (or ``(well, value)`` pairs).  Bare value
    arrays are rejected: measurements are never zipped to layout order.
    """

    records: list[tuple[object, object]] = []
    if isinstance(measurements, Mapping):
        records = list(measurements.items())
    elif isinstance(measurements, (list, tuple)):
        for item in measurements:
            if isinstance(item, Mapping):
                if "well" not in item or "value" not in item:
                    raise ImportError_(
                        "measurement record must carry explicit 'well' and 'value'"
                    )
                records.append((item["well"], item["value"]))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                records.append((item[0], item[1]))
            else:
                raise ImportError_(
                    "measurements must be keyed by well; positional value arrays are rejected"
                )
    else:
        raise ImportError_(
            "measurements must be a mapping or a sequence of well/value records"
        )

    well_map = layout.well_map
    result: dict[str, Fraction] = {}
    seen: set[str] = set()
    for well_value, raw_value in records:
        try:
            well = Well.parse(well_value, layout.rows, layout.cols)
        except WellError as exc:
            raise ImportError_(str(exc)) from exc
        name = well.name
        if name in seen:
            raise ImportError_(f"duplicate measurement for well {name}")
        seen.add(name)
        published = well_map.get(name)
        if published is None:
            raise ImportError_(f"well {name} is not part of the published layout")
        if published.blocked or published.role == ROLE_UNUSED:
            raise ImportError_(
                f"measurement for unused/blocked well {name} is not allowed"
            )
        try:
            value = parse_rational(raw_value, field=f"measurement at {name}")
        except RationalError as exc:
            raise ImportError_(str(exc)) from exc
        result[name] = value
    return result


class ReadoutWarehouse:
    """A bounded warehouse of plate layouts and batch readout runs.

    At most ``MAX_PLATES`` distinct plate ids and at most ``MAX_WELLS`` wells
    per plate (enforced by the published layout product).
    """

    def __init__(self, *, max_plates: int = MAX_PLATES) -> None:
        if max_plates < 1 or max_plates > MAX_PLATES:
            raise WarehouseError(f"max_plates must be in 1..{MAX_PLATES}")
        self._max_plates = max_plates
        self._layouts: dict[tuple[str, str], PublishedLayout] = {}
        self._plate_ids: set[str] = set()
        self._runs: dict[str, ReadoutRun] = {}
        self._fits: dict[str, FitReport] = {}
        self._run_counter = itertools.count(1)

    # -- layouts ---------------------------------------------------------------
    def bind_layout(self, layout: PublishedLayout | Mapping) -> PublishedLayout:
        published = _coerce_layout(layout)
        if published.plate_id not in self._plate_ids and len(self._plate_ids) >= self._max_plates:
            raise WarehouseError(f"warehouse plate limit ({self._max_plates}) reached")
        key = (published.plate_id, published.digest)
        existing = self._layouts.get(key)
        if existing is not None:
            if existing.to_dict() != published.to_dict():
                raise ImportError_("same digest but different layout content")
            return existing
        self._layouts[key] = published
        self._plate_ids.add(published.plate_id)
        return published

    def layout(self, plate_id: str, digest: str) -> PublishedLayout:
        try:
            return self._layouts[(plate_id, digest)]
        except KeyError as exc:
            raise WarehouseError(f"unknown layout {plate_id}/{digest}") from exc

    # -- imports ---------------------------------------------------------------
    def import_run(
        self,
        layout: PublishedLayout | Mapping,
        measurements: object,
        *,
        run_id: str | None = None,
        note: str | None = None,
        expect_digest: str | None = None,
    ) -> ReadoutRun:
        # Validate the run fully before touching warehouse state, so a rejected
        # import cannot consume plate quota or leave an orphan layout behind.
        published = _coerce_layout(layout)
        if expect_digest is not None and published.digest != expect_digest:
            raise ImportError_(
                f"layout digest {published.digest} does not match expected {expect_digest}"
            )
        normalised = _normalise_measurements(published, measurements)
        if run_id is None:
            generated_id = f"run-{next(self._run_counter):04d}"
        elif not isinstance(run_id, str) or not run_id.strip():
            raise ImportError_("run_id must be a non-empty string")
        else:
            generated_id = run_id
        if generated_id in self._runs:
            raise ImportError_(f"duplicate run id {generated_id!r}")
        published = self.bind_layout(published)
        run = ReadoutRun(
            run_id=generated_id,
            layout=published,
            measurements=dict(normalised),
            note=note,
        )
        self._runs[run.run_id] = run
        return run

    def run(self, run_id: str) -> ReadoutRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RunNotFound(run_id) from exc

    def runs(self) -> tuple[ReadoutRun, ...]:
        return tuple(self._runs.values())

    # -- fitting ---------------------------------------------------------------
    def fit(self, run_or_id: ReadoutRun | str) -> FitReport:
        run = self.run(run_or_id) if isinstance(run_or_id, str) else run_or_id
        cached = self._fits.get(run.run_id)
        if cached is not None:
            return cached
        report = self._fit_run(run)
        self._fits[run.run_id] = report
        return report

    def sample_stats(self, run_or_id: ReadoutRun | str, sample_id: str) -> SampleSummary:
        return self.fit(run_or_id).sample(sample_id)

    def _fit_run(self, run: ReadoutRun) -> FitReport:
        layout = run.layout
        structure = _structure_for(layout)
        measured = run.measurements

        used = [w for w in layout.wells if w.role != ROLE_UNUSED]
        observed_control_wells = tuple(
            w.name
            for w in used
            if w.role in (ROLE_BLANK, ROLE_REFERENCE) and w.name in measured
        )
        missing_controls = tuple(
            w.name
            for w in used
            if w.role in (ROLE_BLANK, ROLE_REFERENCE) and w.name not in measured
        )
        factor = _factor_for(layout, observed_control_wells)

        # Target for controls: response - expected = b + r_i + c_j.
        targets: list[Fraction] = []
        for name in observed_control_wells:
            published = layout.well_map[name]
            expected = published.expected if published.expected is not None else Fraction(0)
            targets.append(measured[name] - expected)

        # Solve the (exactly) consistent normal equations via cached transform.
        rhs = [
            sum((factor.design[i][k] * targets[i] for i in range(len(factor.design))), Fraction(0))
            for k in range(factor.n_params)
        ]
        transformed = matvec(factor.transform, rhs) if factor.n_params else []
        particular = [Fraction(0) for _ in range(factor.n_params)]
        for row_index, pivot_col in enumerate(factor.pivots):
            particular[pivot_col] = transformed[row_index]
        # Rows past the rank must have zero transformed right-hand side.
        for row_index in range(factor.rank, factor.n_params):
            if row_index < len(transformed) and transformed[row_index] != 0:
                raise WarehouseError("normal equations inconsistent (internal error)")

        # Residuals and RSS from the original design matrix (independent path).
        predictions = matvec(factor.design, particular) if factor.design else []
        residuals = [pred - target for pred, target in zip(predictions, targets)]
        rss = sum((r * r for r in residuals), Fraction(0))
        max_abs = max((abs(r) for r in residuals), default=Fraction(0))

        effects = _effect_from_params(structure, particular)

        results: list[WellResult] = []
        for published in used:
            feature = _feature(structure, published.row, published.col)
            bias = dot(feature, particular) if feature else Fraction(0)
            deltas = [dot(feature, vec) for vec in factor.null_basis]
            identifiable = all(delta == 0 for delta in deltas)
            proof = None
            proof_direction: Vector | None = None
            if not identifiable:
                witness_index = next(i for i, delta in enumerate(deltas) if delta != 0)
                scale = Fraction(1, 1) / deltas[witness_index]
                proof_direction = [value * scale for value in factor.null_basis[witness_index]]
                effect_delta = _effect_from_params(structure, proof_direction)
                proof = {
                    "direction_index": witness_index,
                    "bias_delta": fmt_rational(Fraction(1)),
                    "delta_b": fmt_rational(effect_delta["b"]),
                    "delta_rows": {
                        str(row): fmt_rational(value)
                        for row, value in effect_delta["rows"].items()
                    },
                    "delta_cols": {
                        str(col): fmt_rational(value)
                        for col, value in effect_delta["cols"].items()
                    },
                    "statement": (
                        "adding this direction to the reported optimal solution keeps every "
                        "observed-control residual unchanged but changes this well's bias by 1"
                    ),
                }

            observed = measured.get(published.name)
            residual = None
            corrected = None
            if observed is not None:
                adjusted = observed - (published.expected or Fraction(0))
                # For controls, measured well coordinates were already used.
                if published.role in (ROLE_BLANK, ROLE_REFERENCE):
                    residual = adjusted - bias
                    status = STATUS_CONTROL
                else:
                    if identifiable:
                        corrected = observed - bias
                        status = STATUS_CORRECTED
                    else:
                        status = STATUS_UNIDENTIFIABLE
            else:
                status = STATUS_MISSING

            results.append(
                WellResult(
                    well=published.name,
                    row=published.row,
                    col=published.col,
                    role=published.role,
                    sample_id=published.sample_id,
                    expected=published.expected,
                    observed=observed,
                    bias=bias,
                    bias_identifiable=identifiable,
                    corrected=corrected,
                    residual=residual,
                    status=status,
                    proof=proof,
                    proof_direction=proof_direction,
                )
            )

        sample_summaries = self._sample_summaries(layout, results)
        verification = self._verify(
            layout, structure, factor, particular, targets, residuals, results, effects, sample_summaries
        )

        report = FitReport(
            run_id=run.run_id,
            plate_id=layout.plate_id,
            layout_version=layout.version,
            layout_digest=layout.digest,
            control_wells=observed_control_wells,
            missing_control_wells=missing_controls,
            observed_control_count=len(observed_control_wells),
            rank=factor.rank,
            n_params=factor.n_params,
            nullity=factor.nullity,
            numerical_rank=factor.numerical_rank,
            rank_tolerance=factor.rank_tolerance,
            residual_sum_of_squares=rss,
            max_abs_control_residual=max_abs,
            parameter_unique=factor.nullity == 0,
            effects=effects,
            wells=tuple(results),
            samples=sample_summaries,
            null_basis=factor.null_basis,
            verification=verification,
        )
        return report

    def _sample_summaries(
        self, layout: PublishedLayout, results: Sequence[WellResult]
    ) -> tuple[SampleSummary, ...]:
        by_name = {w.well: w for w in results}
        summaries: list[SampleSummary] = []
        for sample_id in layout.sample_ids:
            names = layout.sample_wells(sample_id)
            corrected: dict[str, Fraction] = {}
            reasons: list[dict] = []
            for name in names:
                result = by_name[name]
                if result.status == STATUS_CORRECTED and result.corrected is not None:
                    corrected[name] = result.corrected
                else:
                    reason = {
                        "well": name,
                        "status": result.status,
                        "detail": (
                            "no measurement imported for this replicate"
                            if result.status == STATUS_MISSING
                            else "bias is not unique across least-squares minimisers"
                            if result.status == STATUS_UNIDENTIFIABLE
                            else "replicate not correctable"
                        ),
                    }
                    if result.status == STATUS_UNIDENTIFIABLE and result.proof is not None:
                        reason["proof"] = result.proof
                    reasons.append(reason)
            if names and len(corrected) == len(names):
                values = list(corrected.values())
                mean = sum(values, Fraction(0)) / len(values)
                span = max(values) - min(values)
                status = "COMPLETE"
            else:
                mean = None
                span = None
                status = "INCOMPLETE"
            summaries.append(
                SampleSummary(
                    sample_id=sample_id,
                    wells=names,
                    status=status,
                    corrected_values=corrected,
                    mean=mean,
                    range=span,
                    reasons=tuple(reasons),
                )
            )
        return tuple(summaries)

    def _verify(
        self,
        layout: PublishedLayout,
        structure: _Structure,
        factor: _ControlFactor,
        particular: Vector,
        targets: Sequence[Fraction],
        residuals: Sequence[Fraction],
        results: Sequence[WellResult],
        effects: dict[str, object],
        samples: Sequence[SampleSummary],
    ) -> dict[str, object]:
        # 1. Normal equations satisfied.
        normal_residual = [
            sum(
                (factor.design[i][k] * residuals[i] for i in range(len(factor.design))),
                Fraction(0),
            )
            for k in range(factor.n_params)
        ]
        normal_ok = all(value == 0 for value in normal_residual)

        # 2. Effects reproduce every control residual exactly.  The control
        #    target was ``response - expected``, so ``bias = target + residual``.
        well_map = layout.well_map
        reproduced = True
        for name, target, residual in zip(factor.control_wells, targets, residuals):
            published = well_map[name]
            bias = (
                effects["b"]
                + effects["rows"][published.row]
                + effects["cols"][published.col]
            )
            if bias != target + residual:
                reproduced = False
                break

        # 3. Constraints hold.
        row_sum = sum(effects["rows"].values(), Fraction(0))
        col_sum = sum(effects["cols"].values(), Fraction(0))
        constraints_ok = row_sum == 0 and col_sum == 0

        # 4. Null basis really annihilates the control design and has the
        #    reported dimension.
        null_ok = True
        for vec in factor.null_basis:
            if any(
                dot(row, vec) != 0 for row in factor.design
            ):
                null_ok = False
                break
        dimension_ok = len(factor.null_basis) == factor.nullity

        # 5. Identifiability claims agree with direct functional evaluations.
        claims_ok = True
        for result in results:
            feature = _feature(structure, result.row, result.col)
            unique_by_functional = all(dot(feature, vec) == 0 for vec in factor.null_basis)
            if unique_by_functional != result.bias_identifiable:
                claims_ok = False
                break
            if result.status == STATUS_UNIDENTIFIABLE and result.proof_direction is not None:
                # A witness of bias change 1 must not perturb any control row.
                scaled = result.proof_direction
                if any(dot(row, scaled) != 0 for row in factor.design):
                    claims_ok = False
                    break
                if dot(feature, scaled) != 1:
                    claims_ok = False
                    break

        # 6. Sample summaries include every replicate and mean/range are right.
        samples_ok = True
        for summary in samples:
            expected_names = set(layout.sample_wells(summary.sample_id))
            if set(summary.wells) != expected_names:
                samples_ok = False
                break
            if summary.status == "COMPLETE":
                values = list(summary.corrected_values.values())
                if summary.mean != sum(values, Fraction(0)) / len(values):
                    samples_ok = False
                    break
                if summary.range != max(values) - min(values):
                    samples_ok = False
                    break
            elif summary.reasons == () or len(summary.corrected_values) == len(summary.wells):
                samples_ok = False
                break

        numerical_agrees = factor.numerical_rank == factor.rank

        return {
            "normal_equations_zero": normal_ok,
            "control_residuals_reproduced": reproduced,
            "sum_row_effects_zero": row_sum == 0,
            "sum_col_effects_zero": col_sum == 0,
            "nullspace_annihilates_controls": null_ok,
            "nullity_matches_basis": dimension_ok,
            "identifiability_claims_verified": claims_ok,
            "sample_summaries_verified": samples_ok,
            "numerical_rank": factor.numerical_rank,
            "exact_rank": factor.rank,
            "rank_tolerance": factor.rank_tolerance,
            "numerical_rank_agrees": numerical_agrees,
            "ok": all(
                [
                    normal_ok,
                    reproduced,
                    constraints_ok,
                    null_ok,
                    dimension_ok,
                    claims_ok,
                    samples_ok,
                    numerical_agrees,
                ]
            ),
        }

    # -- explanation -----------------------------------------------------------
    def explain(self, run_or_id: ReadoutRun | str) -> dict:
        run = self.run(run_or_id) if isinstance(run_or_id, str) else run_or_id
        report = self.fit(run)
        payload = report.to_dict()
        payload["note"] = run.note
        payload["layout"] = run.layout.to_dict()
        payload["imported_wells"] = sorted(run.measurements)
        payload["missing_used_wells"] = list(run.missing_used_wells)
        return payload
