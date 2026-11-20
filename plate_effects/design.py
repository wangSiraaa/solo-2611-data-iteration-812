"""Immutable, fully-provenanced multi-well plate designs.

A design owns:

* the geometry (at most 8 rows x 12 columns),
* one role per well (blank, reference with a rational expected value,
  sample with a sample id, or unused),
* a set of unavailable ("blocked") wells,
* per-sample replicate-count bounds.

Every mutating operation is a transaction: the candidate state is validated
first and only then swapped in.  A failed replacement therefore leaves the
previous layout untouched, including its version counter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .rational import RationalError, fmt_rational, parse_rational

MAX_ROWS = 8
MAX_COLS = 12
MAX_WELLS = MAX_ROWS * MAX_COLS

ROLE_BLANK = "blank"
ROLE_REFERENCE = "reference"
ROLE_SAMPLE = "sample"
ROLE_UNUSED = "unused"
ROLE_KINDS = (ROLE_BLANK, ROLE_REFERENCE, ROLE_SAMPLE, ROLE_UNUSED)

_WELL_RE = re.compile(r"^([A-Za-z])([0-9]{1,2})$")


class DesignError(ValueError):
    """Base class for design-service errors."""


class WellError(DesignError):
    """Malformed or out-of-plate well coordinate."""


class LayoutValidationError(DesignError):
    """A layout violates one or more design rules."""

    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__(
            "layout rejected: " + "; ".join(report.issues)
            if report.issues
            else "layout rejected"
        )


class PublishedLayoutError(DesignError):
    """An externally supplied published layout is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Coordinates and roles
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Well:
    """A stable zero-based plate coordinate."""

    row: int
    col: int

    def __post_init__(self) -> None:
        if not isinstance(self.row, int) or not isinstance(self.col, int):
            raise WellError("well row/col must be integers")
        if self.row < 0 or self.col < 0:
            raise WellError("negative well coordinate")

    @property
    def name(self) -> str:
        return f"{chr(ord('A') + self.row)}{self.col + 1}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    @classmethod
    def parse(cls, value: object, rows: int, cols: int) -> "Well":
        if isinstance(value, Well):
            well = value
        elif isinstance(value, str):
            match = _WELL_RE.match(value.strip())
            if not match:
                raise WellError(f"malformed well coordinate: {value!r}")
            row = ord(match.group(1).upper()) - ord("A")
            col = int(match.group(2)) - 1
            well = cls(row, col)
        else:
            raise WellError(f"unsupported well coordinate type: {type(value).__name__}")
        if well.row >= rows or well.col >= cols:
            raise WellError(
                f"well {well.name} is outside the {rows}x{cols} plate"
            )
        return well


@dataclass(frozen=True)
class Role:
    """A well role.

    ``blank`` has expected value 0, ``reference`` carries the stated rational
    expectation, ``sample`` carries a sample id, ``unused`` carries nothing.
    """

    kind: str
    expected: Fraction | None = None
    sample_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ROLE_KINDS:
            raise DesignError(f"unknown role kind {self.kind!r}")
        if self.kind == ROLE_BLANK:
            expected = Fraction(0) if self.expected is None else parse_rational(self.expected, field="blank expected")
            if expected != 0:
                raise DesignError("blank wells have expected value 0 by definition")
            object.__setattr__(self, "expected", Fraction(0))
            object.__setattr__(self, "sample_id", None)
        elif self.kind == ROLE_REFERENCE:
            if self.expected is None:
                raise DesignError("reference role requires an expected value")
            object.__setattr__(self, "expected", parse_rational(self.expected, field="reference expected"))
            object.__setattr__(self, "sample_id", None)
        elif self.kind == ROLE_SAMPLE:
            if not isinstance(self.sample_id, str) or not self.sample_id.strip():
                raise DesignError("sample role requires a non-empty sample id")
            object.__setattr__(self, "sample_id", self.sample_id.strip())
            object.__setattr__(self, "expected", None)
        else:  # unused
            object.__setattr__(self, "expected", None)
            object.__setattr__(self, "sample_id", None)

    # Convenience constructors -------------------------------------------------
    @staticmethod
    def blank() -> "Role":
        return Role(ROLE_BLANK)

    @staticmethod
    def reference(expected: object) -> "Role":
        return Role(ROLE_REFERENCE, expected=parse_rational(expected, field="reference expected"))

    @staticmethod
    def sample(sample_id: str) -> "Role":
        return Role(ROLE_SAMPLE, sample_id=sample_id)

    @staticmethod
    def unused() -> "Role":
        return Role(ROLE_UNUSED)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleCheck:
    sample_id: str
    wells: tuple[str, ...]
    replicate_count: int
    min_replicates: int
    max_replicates: int
    rows: tuple[int, ...]
    row_unique: bool
    count_in_range: bool
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    issues: tuple[str, ...]
    samples: tuple[SampleCheck, ...]

    def sample(self, sample_id: str) -> SampleCheck:
        for check in self.samples:
            if check.sample_id == sample_id:
                return check
        raise KeyError(sample_id)

    def describe(self) -> str:
        if self.ok:
            return "layout valid"
        return "layout invalid: " + "; ".join(self.issues)


# ---------------------------------------------------------------------------
# Published (immutable) layout product
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedWell:
    name: str
    row: int
    col: int
    role: str
    expected: Fraction | None
    sample_id: str | None
    blocked: bool
    provenance: str

    def to_dict(self) -> dict:
        return {
            "well": self.name,
            "row": self.row,
            "col": self.col,
            "role": self.role,
            "expected": None if self.expected is None else fmt_rational(self.expected),
            "sample_id": self.sample_id,
            "blocked": self.blocked,
            "provenance": self.provenance,
        }

    def semantic_record(self) -> list[object]:
        return [
            self.name,
            self.row,
            self.col,
            self.role,
            None if self.expected is None else fmt_rational(self.expected),
            self.sample_id,
            bool(self.blocked),
        ]


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _layout_semantics(
    plate_id: str,
    rows: int,
    cols: int,
    version: int,
    min_replicates: int,
    max_replicates: int,
    sample_bounds: Sequence[tuple[str, int, int]],
    wells: Sequence[PublishedWell],
) -> dict:
    return {
        "format": "plate-layout-semantics/v1",
        "plate_id": plate_id,
        "rows": rows,
        "cols": cols,
        "version": version,
        "min_replicates": min_replicates,
        "max_replicates": max_replicates,
        "sample_bounds": [[sid, lo, hi] for sid, lo, hi in sorted(sample_bounds)],
        "wells": [w.semantic_record() for w in sorted(wells, key=lambda w: (w.row, w.col))],
    }


def layout_digest(payload_semantics: Mapping) -> str:
    return hashlib.sha256(_canonical_json(payload_semantics).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublishedLayout:
    """The product handed to the readout warehouse.

    Every well of the plate is present, including unused and blocked wells.
    Blank wells are explicit observations with expected value zero; they are
    *not* "no data".
    """

    plate_id: str
    rows: int
    cols: int
    version: int
    digest: str
    wells: tuple[PublishedWell, ...]
    min_replicates: int
    max_replicates: int
    sample_bounds: tuple[tuple[str, int, int], ...]

    # -- lookups ---------------------------------------------------------------
    @property
    def well_names(self) -> tuple[str, ...]:
        return tuple(w.name for w in self.wells)

    @property
    def well_map(self) -> dict[str, PublishedWell]:
        return {w.name: w for w in self.wells}

    @property
    def blocked_wells(self) -> tuple[str, ...]:
        return tuple(w.name for w in self.wells if w.blocked)

    @property
    def control_wells(self) -> tuple[str, ...]:
        return tuple(w.name for w in self.wells if w.role in (ROLE_BLANK, ROLE_REFERENCE))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        ids = {w.sample_id for w in self.wells if w.role == ROLE_SAMPLE}
        return tuple(sorted(ids))

    @property
    def used_wells(self) -> tuple[str, ...]:
        return tuple(w.name for w in self.wells if w.role != ROLE_UNUSED)

    def well(self, value: object) -> PublishedWell:
        name = Well.parse(value, self.rows, self.cols).name
        return self.well_map[name]

    def sample_wells(self, sample_id: str) -> tuple[str, ...]:
        return tuple(
            w.name
            for w in self.wells
            if w.role == ROLE_SAMPLE and w.sample_id == sample_id
        )

    def bounds_for(self, sample_id: str) -> tuple[int, int]:
        for sid, lo, hi in self.sample_bounds:
            if sid == sample_id:
                return lo, hi
        return self.min_replicates, self.max_replicates

    # -- serialisation ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "plate-layout/v1",
            "plate_id": self.plate_id,
            "rows": self.rows,
            "cols": self.cols,
            "version": self.version,
            "digest": self.digest,
            "min_replicates": self.min_replicates,
            "max_replicates": self.max_replicates,
            "sample_bounds": [
                {"sample_id": sid, "min_replicates": lo, "max_replicates": hi}
                for sid, lo, hi in self.sample_bounds
            ],
            "wells": [w.to_dict() for w in sorted(self.wells, key=lambda w: (w.row, w.col))],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "PublishedLayout":
        try:
            if payload.get("format") != "plate-layout/v1":
                raise PublishedLayoutError("not a plate-layout/v1 product")
            plate_id = payload["plate_id"]
            rows = int(payload["rows"])
            cols = int(payload["cols"])
            version = int(payload["version"])
            digest = str(payload["digest"])
            min_rep = int(payload["min_replicates"])
            max_rep = int(payload["max_replicates"])
            raw_bounds = payload.get("sample_bounds", [])
            raw_wells = payload["wells"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PublishedLayoutError(f"malformed layout product: {exc}") from exc

        if not isinstance(plate_id, str) or not plate_id:
            raise PublishedLayoutError("plate_id must be a non-empty string")
        if not 1 <= rows <= MAX_ROWS or not 1 <= cols <= MAX_COLS:
            raise PublishedLayoutError(f"invalid plate geometry {rows}x{cols}")
        if not isinstance(raw_wells, list):
            raise PublishedLayoutError("wells must be a list")

        sample_bounds: list[tuple[str, int, int]] = []
        for item in raw_bounds:
            try:
                sid = str(item["sample_id"])
                lo = int(item["min_replicates"])
                hi = int(item["max_replicates"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PublishedLayoutError(f"malformed sample bound: {item!r}") from exc
            if not sid or lo < 1 or hi < lo or hi > rows:
                raise PublishedLayoutError(f"invalid sample bound for {sid!r}")
            sample_bounds.append((sid, lo, hi))

        wells: list[PublishedWell] = []
        seen: set[tuple[int, int]] = set()
        for item in raw_wells:
            try:
                name = str(item["well"])
                row = int(item["row"])
                col = int(item["col"])
                role = str(item["role"])
                expected_raw = item["expected"]
                sample_id = item["sample_id"]
                blocked = bool(item["blocked"])
                provenance = str(item.get("provenance", "external"))
            except (KeyError, TypeError, ValueError) as exc:
                raise PublishedLayoutError(f"malformed well record: {item!r}") from exc
            if (row, col) in seen:
                raise PublishedLayoutError(f"duplicate well position {row},{col}")
            seen.add((row, col))
            if not (0 <= row < rows and 0 <= col < cols):
                raise PublishedLayoutError(f"well {name} outside plate")
            expected_well = Well(row, col).name
            if name != expected_well:
                raise PublishedLayoutError(
                    f"well name/position mismatch: {name} at ({row},{col})"
                )
            if role not in ROLE_KINDS:
                raise PublishedLayoutError(f"unknown role {role!r} at {name}")
            expected: Fraction | None = None
            if role == ROLE_REFERENCE:
                if expected_raw is None:
                    raise PublishedLayoutError(f"reference well {name} lacks expected value")
                try:
                    expected = parse_rational(expected_raw, field=f"well {name} expected")
                except RationalError as exc:
                    raise PublishedLayoutError(str(exc)) from exc
            elif role == ROLE_BLANK:
                # Blank expected values are published explicitly as 0.
                if expected_raw is not None:
                    try:
                        expected = parse_rational(expected_raw, field=f"well {name} expected")
                    except RationalError as exc:
                        raise PublishedLayoutError(str(exc)) from exc
                    if expected != 0:
                        raise PublishedLayoutError(
                            f"blank well {name} must have expected value 0"
                        )
                expected = Fraction(0)
            elif expected_raw is not None:
                raise PublishedLayoutError(f"non-reference well {name} carries an expected value")
            if role == ROLE_SAMPLE:
                if not isinstance(sample_id, str) or not sample_id:
                    raise PublishedLayoutError(f"sample well {name} lacks sample id")
            elif sample_id is not None:
                raise PublishedLayoutError(f"non-sample well {name} carries a sample id")
            if blocked and role != ROLE_UNUSED:
                raise PublishedLayoutError(f"blocked well {name} must be unused")
            wells.append(
                PublishedWell(name, row, col, role, expected, sample_id, blocked, provenance)
            )

        if len(wells) != rows * cols or len(seen) != rows * cols:
            raise PublishedLayoutError(
                f"layout product must list all {rows * cols} well positions exactly once"
            )

        # Structural sample checks: published products must not contain a sample
        # whose declared replicate structure is impossible.
        by_sample: dict[str, list[int]] = {}
        for w in wells:
            if w.role == ROLE_SAMPLE:
                by_sample.setdefault(w.sample_id, []).append(w.row)
        bound_map = {sid: (lo, hi) for sid, lo, hi in sample_bounds}
        for sid, rws in by_sample.items():
            lo, hi = bound_map.get(sid, (min_rep, max_rep))
            if not (lo <= len(rws) <= hi):
                raise PublishedLayoutError(
                    f"sample {sid!r} has {len(rws)} replicates outside [{lo},{hi}]"
                )
            if len(set(rws)) != len(rws):
                raise PublishedLayoutError(f"sample {sid!r} repeats a row")

        published = cls(
            plate_id=plate_id,
            rows=rows,
            cols=cols,
            version=version,
            digest=digest,
            wells=tuple(sorted(wells, key=lambda w: (w.row, w.col))),
            min_replicates=min_rep,
            max_replicates=max_rep,
            sample_bounds=tuple(sorted(sample_bounds)),
        )
        computed = layout_digest(
            _layout_semantics(
                plate_id, rows, cols, version, min_rep, max_rep, sample_bounds, published.wells
            )
        )
        if computed != digest:
            raise PublishedLayoutError(
                f"layout digest mismatch: product says {digest}, content hashes to {computed}"
            )
        return published


# ---------------------------------------------------------------------------
# Mutable design service
# ---------------------------------------------------------------------------


class PlateDesign:
    """A transactional design service for one plate.

    Operations either commit a fully validated layout and bump the version, or
    raise and leave the previous layout untouched.
    """

    def __init__(
        self,
        plate_id: str,
        rows: int = MAX_ROWS,
        cols: int = MAX_COLS,
        *,
        min_replicates: int = 1,
        max_replicates: int | None = None,
        roles: Mapping[object, Role] | None = None,
        blocked: Iterable[object] = (),
        sample_bounds: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        if not isinstance(plate_id, str) or not plate_id.strip():
            raise DesignError("plate_id must be a non-empty string")
        if not isinstance(rows, int) or not isinstance(cols, int):
            raise DesignError("rows/cols must be integers")
        if not 1 <= rows <= MAX_ROWS:
            raise DesignError(f"rows must be in 1..{MAX_ROWS}")
        if not 1 <= cols <= MAX_COLS:
            raise DesignError(f"cols must be in 1..{MAX_COLS}")
        if rows * cols > MAX_WELLS:
            raise DesignError(f"plate exceeds {MAX_WELLS} wells")
        max_rep = rows if max_replicates is None else max_replicates
        if not isinstance(min_replicates, int) or not isinstance(max_rep, int):
            raise DesignError("replicate bounds must be integers")
        if min_replicates < 1 or max_rep < min_replicates or max_rep > rows:
            raise DesignError(
                f"replicate bounds must satisfy 1 <= min <= max <= rows ({rows})"
            )

        self._plate_id = plate_id.strip()
        self._rows = rows
        self._cols = cols
        self._min_replicates = min_replicates
        self._max_replicates = max_rep
        self._roles: dict[Well, Role] = {}
        self._blocked: set[Well] = set()
        self._origins: dict[Well, str] = {}
        self._sample_bounds: dict[str, tuple[int, int]] = {}
        self._version = 1

        for well_value, role in (roles or {}).items():
            well = Well.parse(well_value, rows, cols)
            if not isinstance(role, Role):
                raise DesignError(f"role for {well.name} must be a Role")
            if role.kind != ROLE_UNUSED:
                self._roles[well] = role
                self._origins[well] = "initial"
        for well_value in blocked:
            well = Well.parse(well_value, rows, cols)
            self._blocked.add(well)
            self._origins[well] = "initial_blocked"
        for sid, (lo, hi) in (sample_bounds or {}).items():
            self._sample_bounds[sid] = (int(lo), int(hi))
        self._validate_or_raise()

    # -- introspection ---------------------------------------------------------
    @property
    def plate_id(self) -> str:
        return self._plate_id

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def version(self) -> int:
        return self._version

    @property
    def min_replicates(self) -> int:
        return self._min_replicates

    @property
    def max_replicates(self) -> int:
        return self._max_replicates

    @property
    def roles(self) -> dict[Well, Role]:
        return dict(self._roles)

    @property
    def blocked(self) -> frozenset[Well]:
        return frozenset(self._blocked)

    @property
    def sample_bounds(self) -> dict[str, tuple[int, int]]:
        return dict(self._sample_bounds)

    def role_of(self, well_value: object) -> Role:
        well = Well.parse(well_value, self._rows, self._cols)
        return self._roles.get(well, Role.unused())

    def all_wells(self) -> tuple[Well, ...]:
        return tuple(Well(r, c) for r in range(self._rows) for c in range(self._cols))

    def available_wells(self, sample_id: str | None = None) -> tuple[Well, ...]:
        """Wells that may still be assigned to ``sample_id``.

        Excludes unavailable wells, wells carrying another role, and rows that
        already contain a replicate of this sample.  Returns an empty tuple once
        the sample has reached its maximum replicate count.
        """

        if sample_id is not None:
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise DesignError("sample_id must be a non-empty string")
            sample_id = sample_id.strip()
            existing = [w for w, r in self._roles.items() if r.kind == ROLE_SAMPLE and r.sample_id == sample_id]
            lo, hi = self._sample_bounds.get(sample_id, (self._min_replicates, self._max_replicates))
            if len(existing) >= hi:
                return ()
            used_rows = {w.row for w in existing}
        else:
            used_rows = set()
        return tuple(
            w
            for w in self.all_wells()
            if w not in self._roles and w not in self._blocked and w.row not in used_rows
        )

    # -- validation ------------------------------------------------------------
    def _validate_state(
        self,
        roles: Mapping[Well, Role],
        blocked: set[Well],
        bounds: Mapping[str, tuple[int, int]],
        global_bounds: tuple[int, int] | None = None,
    ) -> ValidationReport:
        default_min, default_max = global_bounds or (
            self._min_replicates,
            self._max_replicates,
        )
        issues: list[str] = []
        for well, role in roles.items():
            if not (0 <= well.row < self._rows and 0 <= well.col < self._cols):
                issues.append(f"{well.name}: outside plate")
            if role.kind not in ROLE_KINDS:
                issues.append(f"{well.name}: unknown role {role.kind}")
            if role.kind == ROLE_REFERENCE and role.expected is None:
                issues.append(f"{well.name}: reference without expected value")
            if role.kind == ROLE_BLANK and role.expected != 0:
                issues.append(f"{well.name}: blank expected value must be 0")
            if role.kind == ROLE_SAMPLE and not role.sample_id:
                issues.append(f"{well.name}: sample without sample id")
            if well in blocked and role.kind != ROLE_UNUSED:
                issues.append(
                    f"{well.name}: blocked/unavailable well must be unused, found {role.kind}"
                )

        for well in blocked:
            if not (0 <= well.row < self._rows and 0 <= well.col < self._cols):
                issues.append(f"{well.name}: blocked well outside plate")

        for sid, (lo, hi) in bounds.items():
            if not isinstance(sid, str) or not sid.strip():
                issues.append("sample bound with empty sample id")
            if not isinstance(lo, int) or not isinstance(hi, int):
                issues.append(f"sample {sid!r}: replicate bounds must be integers")
            elif lo < 1 or hi < lo or hi > self._rows:
                issues.append(
                    f"sample {sid!r}: invalid replicate bounds [{lo},{hi}] for {self._rows} rows"
                )

        grouped: dict[str, list[Well]] = {}
        for well, role in roles.items():
            if role.kind == ROLE_SAMPLE:
                grouped.setdefault(role.sample_id, []).append(well)

        checks: list[SampleCheck] = []
        for sid in sorted(grouped):
            wells = tuple(sorted(grouped[sid]))
            rows = tuple(w.row for w in wells)
            lo, hi = bounds.get(sid, (default_min, default_max))
            count_ok = lo <= len(wells) <= hi
            row_unique = len(set(rows)) == len(rows)
            sample_issues: list[str] = []
            if not count_ok:
                sample_issues.append(
                    f"sample {sid!r}: replicate count {len(wells)} outside [{lo},{hi}]"
                )
            if not row_unique:
                repeated = sorted({r for r in rows if rows.count(r) > 1})
                repeated_names = [
                    w.name for w in wells if w.row in repeated
                ]
                sample_issues.append(
                    f"sample {sid!r}: replicates share row(s) {repeated} ({', '.join(repeated_names)})"
                )
            issues.extend(sample_issues)
            checks.append(
                SampleCheck(
                    sample_id=sid,
                    wells=tuple(w.name for w in wells),
                    replicate_count=len(wells),
                    min_replicates=lo,
                    max_replicates=hi,
                    rows=rows,
                    row_unique=row_unique,
                    count_in_range=count_ok,
                    issues=tuple(sample_issues),
                )
            )

        # Note: a declared bound for a sample with no replicate is kept as
        # metadata.  It becomes binding again as soon as replicates exist, so
        # clearing all replicates of a sample remains a legal edit.

        return ValidationReport(ok=not issues, issues=tuple(issues), samples=tuple(checks))

    def validate(self) -> ValidationReport:
        return self._validate_state(self._roles, self._blocked, self._sample_bounds)

    def _validate_or_raise(
        self,
        roles: Mapping[Well, Role] | None = None,
        blocked: set[Well] | None = None,
        bounds: Mapping[str, tuple[int, int]] | None = None,
    ) -> ValidationReport:
        report = self._validate_state(
            self._roles if roles is None else roles,
            self._blocked if blocked is None else blocked,
            self._sample_bounds if bounds is None else bounds,
        )
        if not report.ok:
            raise LayoutValidationError(report)
        return report

    # -- transactions ----------------------------------------------------------
    def _commit(
        self,
        roles: dict[Well, Role],
        blocked: set[Well],
        bounds: dict[str, tuple[int, int]],
        origins: dict[Well, str],
    ) -> None:
        report = self._validate_state(roles, blocked, bounds)
        if not report.ok:
            raise LayoutValidationError(report)
        self._roles = roles
        self._blocked = blocked
        self._sample_bounds = bounds
        self._origins = origins
        self._version += 1

    def _copy_state(self) -> tuple[dict[Well, Role], set[Well], dict[str, tuple[int, int]], dict[Well, str]]:
        return (dict(self._roles), set(self._blocked), dict(self._sample_bounds), dict(self._origins))

    def _set_role_atomic(self, wells: Sequence[object], role: Role, origin: str) -> None:
        roles, blocked, bounds, origins = self._copy_state()
        for value in wells:
            well = Well.parse(value, self._rows, self._cols)
            if role.kind == ROLE_UNUSED:
                roles.pop(well, None)
            else:
                roles[well] = role
            origins[well] = origin
        self._commit(roles, blocked, bounds, origins)

    def set_role(self, well_value: object, role: Role) -> None:
        if not isinstance(role, Role):
            raise DesignError("role must be a Role")
        self._set_role_atomic([well_value], role, f"set_role:{role.kind}")

    def set_blank(self, well_values: Iterable[object]) -> None:
        self._set_role_atomic(list(well_values), Role.blank(), "set_blank")

    def set_reference(self, well_values: Iterable[object], expected: object) -> None:
        role = Role.reference(expected)
        self._set_role_atomic(list(well_values), role, "set_reference")

    def correct_reference(self, well_values: Iterable[object], expected: object) -> None:
        """Alias that makes the intent explicit: a correction is a new version."""

        self.set_reference(well_values, expected)

    def define_sample(self, sample_id: str, well_values: Iterable[object]) -> None:
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise DesignError("sample_id must be a non-empty string")
        sample_id = sample_id.strip()
        targets = [Well.parse(v, self._rows, self._cols) for v in well_values]
        if len(set(targets)) != len(targets):
            raise DesignError(f"sample {sample_id!r}: duplicate target wells")
        roles, blocked, bounds, origins = self._copy_state()
        for well, role in list(roles.items()):
            if role.kind == ROLE_SAMPLE and role.sample_id == sample_id:
                del roles[well]
                origins[well] = f"define_sample:{sample_id}:released"
        for well in targets:
            if well in blocked:
                raise LayoutValidationError(
                    ValidationReport(
                        ok=False,
                        issues=(f"{well.name}: blocked/unavailable well cannot host a sample",),
                        samples=(),
                    )
                )
            roles[well] = Role.sample(sample_id)
            origins[well] = f"define_sample:{sample_id}"
        self._commit(roles, blocked, bounds, origins)

    def clear_wells(self, well_values: Iterable[object]) -> None:
        self._set_role_atomic(list(well_values), Role.unused(), "cleared")

    def block_wells(self, well_values: Iterable[object]) -> None:
        roles, blocked, bounds, origins = self._copy_state()
        for value in well_values:
            well = Well.parse(value, self._rows, self._cols)
            current = roles.get(well)
            if current is not None and current.kind != ROLE_UNUSED:
                raise LayoutValidationError(
                    ValidationReport(
                        ok=False,
                        issues=(
                            f"{well.name}: cannot block a well that carries role {current.kind}; clear it first",
                        ),
                        samples=(),
                    )
                )
            blocked.add(well)
            origins[well] = "blocked"
        self._commit(roles, blocked, bounds, origins)

    def unblock_wells(self, well_values: Iterable[object]) -> None:
        roles, blocked, bounds, origins = self._copy_state()
        for value in well_values:
            well = Well.parse(value, self._rows, self._cols)
            blocked.discard(well)
            origins[well] = "unblocked"
        self._commit(roles, blocked, bounds, origins)

    def set_sample_bounds(self, sample_id: str, min_replicates: int, max_replicates: int) -> None:
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise DesignError("sample_id must be a non-empty string")
        roles, blocked, bounds, origins = self._copy_state()
        bounds[sample_id.strip()] = (min_replicates, max_replicates)
        self._commit(roles, blocked, bounds, origins)

    def replace_layout(
        self,
        roles: Mapping[object, Role],
        *,
        blocked: Iterable[object] = (),
        sample_bounds: Mapping[str, tuple[int, int]] | None = None,
        min_replicates: int | None = None,
        max_replicates: int | None = None,
    ) -> None:
        """Replace the whole layout atomically.

        Any validation failure raises ``LayoutValidationError`` and leaves the
        current layout *and* version untouched.
        """

        new_roles: dict[Well, Role] = {}
        origins: dict[Well, str] = {}
        for value, role in roles.items():
            if not isinstance(role, Role):
                raise DesignError(f"role for {value!r} must be a Role")
            well = Well.parse(value, self._rows, self._cols)
            if well in new_roles:
                raise DesignError(f"duplicate well {well.name} in replacement layout")
            if role.kind != ROLE_UNUSED:
                new_roles[well] = role
            origins[well] = "replace_layout"
        new_blocked: set[Well] = set()
        for value in blocked:
            well = Well.parse(value, self._rows, self._cols)
            new_blocked.add(well)
            origins.setdefault(well, "replace_layout:blocked")
        new_bounds = dict(self._sample_bounds) if sample_bounds is None else {
            str(sid): (int(lo), int(hi)) for sid, (lo, hi) in sample_bounds.items()
        }
        new_min = self._min_replicates if min_replicates is None else int(min_replicates)
        new_max = self._max_replicates if max_replicates is None else int(max_replicates)
        if not 1 <= new_min <= new_max <= self._rows:
            raise LayoutValidationError(
                ValidationReport(
                    ok=False,
                    issues=(f"invalid replicate bounds [{new_min},{new_max}]",),
                    samples=(),
                )
            )
        report = self._validate_state(
            new_roles, new_blocked, new_bounds, global_bounds=(new_min, new_max)
        )
        if not report.ok:
            raise LayoutValidationError(report)
        self._roles = new_roles
        self._blocked = new_blocked
        self._sample_bounds = new_bounds
        self._origins = origins
        self._min_replicates = new_min
        self._max_replicates = new_max
        self._version += 1

    # -- publishing ------------------------------------------------------------
    def publish(self) -> PublishedLayout:
        report = self.validate()
        if not report.ok:
            raise LayoutValidationError(report)
        wells: list[PublishedWell] = []
        for well in self.all_wells():
            role = self._roles.get(well, Role.unused())
            blocked = well in self._blocked
            if blocked:
                provenance = self._origins.get(well, "blocked")
            elif role.kind == ROLE_UNUSED:
                provenance = self._origins.get(well, "unassigned")
            else:
                provenance = self._origins.get(well, "explicit")
            wells.append(
                PublishedWell(
                    name=well.name,
                    row=well.row,
                    col=well.col,
                    role=role.kind,
                    expected=role.expected,
                    sample_id=role.sample_id,
                    blocked=blocked,
                    provenance=provenance,
                )
            )
        semantics = _layout_semantics(
            self._plate_id,
            self._rows,
            self._cols,
            self._version,
            self._min_replicates,
            self._max_replicates,
            list(self._sample_bounds.items()),
            wells,
        )
        return PublishedLayout(
            plate_id=self._plate_id,
            rows=self._rows,
            cols=self._cols,
            version=self._version,
            digest=layout_digest(semantics),
            wells=tuple(wells),
            min_replicates=self._min_replicates,
            max_replicates=self._max_replicates,
            sample_bounds=tuple(sorted(self._sample_bounds.items())),
        )


def gcd_list(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, value)
    return result
