"""Offline reader for unseparated scan bands, driven by a glyph font file.

Input
-----
* a :class:`~plate_effects.font_builder.GlyphFont` (loaded from its exported
  JSON file, so reading works fully offline and independently of the runs),
* the exact label length ``k``,
* one scan band: full column vectors of ``font.height`` 0/1 rows, with no
  masking and no inserted columns, and
* a :class:`LabelContract`: per-position integer weights, a modulus and the
  allowed remainder set.  A digit string is valid iff
  ``sum(weight_i * digit_i) mod modulus`` is an allowed remainder; leading
  zeros count as ordinary characters.

The band has no separators, so a solution is a partition of *all* columns into
exactly ``k`` glyph-width slices, each matching a font variant of the digit at
that position.  Slicing must start at column 0 and consume the final column:
there is no slack and no arbitrary scan window.

Exact dynamic programming
-------------------------
A state is ``(position, residue)`` at depth ``i`` (i characters consumed).
For every reachable state we keep at most two predecessor entries, each
pointing to one predecessor state and one whole-glyph witness (a single font
variant and its origin).  Keeping two is exact, not a heuristic:

* distinct predecessors of the same state differ in position, hence in length,
  so they already yield distinct strings;
* once two are stored, every later feasibility fact depends only on the
  remaining suffix and on ``(position, residue)``, never on how the prefix was
  built, so a third prefix can never be needed to answer "no solution / unique
  / two strings".

Long runs of identical columns therefore create no combinatorial explosion and
no attempt-count guessing: work is bounded by
``k * n_columns * modulus * 10 * max_width``.  The same digit string produced
via several sheets, thresholds or cuts is counted once; only genuinely
different digit strings can make the answer ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .font_builder import GlyphFont, GlyphOrigin, GlyphVariant

MAX_LABEL_DIGITS = 80
MAX_BAND_COLUMNS = 480
MAX_MODULUS = 97

STATUS_NO_SOLUTION = "no_solution"
STATUS_UNIQUE = "unique"
STATUS_AMBIGUOUS = "ambiguous"

# Keep exactly two witnesses per DP state; see module docstring for why two is
# an exact bound rather than an arbitrary cap.
_STATE_CAP = 2


class LabelReaderError(ValueError):
    """Raised when a band or contract violates the reader contract."""


# ---------------------------------------------------------------------------
# Contract and result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelContract:
    """Checksum contract of one label batch.

    ``weights[i]`` is the integer weight of label position ``i``.  A digit
    string ``d_0..d_{k-1}`` passes iff
    ``sum(weights[i] * d_i) mod modulus`` belongs to ``allowed_remainders``.
    """

    weights: tuple[int, ...]
    modulus: int
    allowed_remainders: frozenset[int]
    name: str = "batch-contract"

    def __post_init__(self) -> None:
        if not self.weights:
            raise LabelReaderError("contract needs at least one weight")
        if len(self.weights) > MAX_LABEL_DIGITS:
            raise LabelReaderError(f"at most {MAX_LABEL_DIGITS} positions are supported")
        if not 1 <= self.modulus <= MAX_MODULUS:
            raise LabelReaderError(f"modulus must be in 1..{MAX_MODULUS}")
        if not self.allowed_remainders:
            raise LabelReaderError("at least one allowed remainder is required")
        bad = [r for r in self.allowed_remainders if not 0 <= r < self.modulus]
        if bad:
            raise LabelReaderError(f"remainders must lie in 0..modulus-1; bad: {bad}")

    @classmethod
    def create(
        cls,
        weights: Sequence[int],
        modulus: int,
        allowed_remainders: Sequence[int],
        *,
        name: str = "batch-contract",
    ) -> "LabelContract":
        return cls(
            weights=tuple(int(w) for w in weights),
            modulus=int(modulus),
            allowed_remainders=frozenset(int(r) for r in allowed_remainders),
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weights": list(self.weights),
            "modulus": self.modulus,
            "allowed_remainders": sorted(self.allowed_remainders),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "LabelContract":
        return cls.create(
            payload["weights"],
            payload["modulus"],
            payload["allowed_remainders"],
            name=payload.get("name", "batch-contract"),
        )


@dataclass(frozen=True)
class CharacterWitness:
    """Per-character slice and the verifiable whole-glyph source behind it."""

    position: int
    digit: str
    column_start: int
    column_end: int  # exclusive
    columns: tuple[tuple[int, ...], ...]
    variant_id: str
    origin: GlyphOrigin

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "digit": self.digit,
            "column_interval": [self.column_start, self.column_end],
            "columns": [list(column) for column in self.columns],
            "variant_id": self.variant_id,
            "origin": self.origin.to_dict(),
        }


@dataclass(frozen=True)
class LabelWitness:
    """One complete digit string with every character's slice and source."""

    digits: str
    checksum: int
    characters: tuple[CharacterWitness, ...]

    def to_dict(self) -> dict:
        return {
            "digits": self.digits,
            "checksum": self.checksum,
            "characters": [char.to_dict() for char in self.characters],
        }


@dataclass(frozen=True)
class ReadResult:
    """Outcome of one band read: no solution, one string or two strings."""

    status: str
    k: int
    n_columns: int
    contract_name: str
    witnesses: tuple[LabelWitness, ...] = field(default=())

    @property
    def digits(self) -> tuple[str, ...]:
        return tuple(w.digits for w in self.witnesses)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "k": self.k,
            "n_columns": self.n_columns,
            "contract": self.contract_name,
            "witness_count": len(self.witnesses),
            "witnesses": [witness.to_dict() for witness in self.witnesses],
        }


# ---------------------------------------------------------------------------
# DP internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pointer:
    """One predecessor edge of a DP state (a whole-glyph match)."""

    prev_position: int
    prev_residue: int
    digit: str
    start: int
    end: int
    variant: GlyphVariant


@dataclass(frozen=True)
class _Prefix:
    """One distinct digit prefix reaching a DP state, with one witness edge."""

    digits: tuple[str, ...]
    pointer: "_Pointer | None"  # None only for the empty prefix at depth 0


@dataclass(frozen=True)
class _LayerEntry:
    """Up to two distinct prefix strings reaching one ``(position, residue)``.

    Identical prefixes produced by different cuts, sheets or thresholds are
    stored once; the first witness is retained.
    """

    prefixes: tuple[_Prefix, ...]


def _normalise_band(
    band: Sequence[Sequence[int]], height: int
) -> tuple[tuple[int, ...], ...]:
    columns: list[tuple[int, ...]] = []
    for index, raw_column in enumerate(band):
        column = tuple(int(bit) for bit in raw_column)
        if len(column) != height:
            raise LabelReaderError(
                f"band column {index} has height {len(column)}, font height is {height}"
            )
        if any(bit not in (0, 1) for bit in column):
            raise LabelReaderError(f"band column {index} contains a non-binary cell")
        columns.append(column)
    return tuple(columns)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class LabelInterpreter:
    """Offline, font-file-driven reader for unseparated scan bands."""

    def __init__(self, font: GlyphFont) -> None:
        self.font = font
        self.height = font.height
        # digit -> (width, variants); missing digits are simply absent.
        self._digits: dict[str, tuple[int, tuple[GlyphVariant, ...]]] = {}
        for digit, variants in font.variants.items():
            if variants:
                width = variants[0].width
                if any(v.width != width or v.height != self.height for v in variants):
                    raise LabelReaderError(f"font digit {digit} has inconsistent geometry")
                self._digits[digit] = (width, variants)
        if not self._digits:
            raise LabelReaderError("font contains no usable glyph variants")

    @classmethod
    def from_file(cls, path: str) -> "LabelInterpreter":
        return cls(GlyphFont.load(path))

    # -- matching --------------------------------------------------------------
    def _matches_at(
        self, band: Sequence[tuple[int, ...]], start: int, variant: GlyphVariant
    ) -> bool:
        end = start + variant.width
        if end > len(band):
            return False
        return all(band[start + offset] == variant.columns[offset] for offset in range(variant.width))

    def solve(
        self, band: Sequence[Sequence[int]], k: int, contract: LabelContract
    ) -> ReadResult:
        """Read one band under one batch contract.

        The same interpreter instance may re-read the same band with a
        different contract: only the contract argument changes.
        """

        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_LABEL_DIGITS:
            raise LabelReaderError(f"k must be an integer in 1..{MAX_LABEL_DIGITS}")
        if len(contract.weights) != k:
            raise LabelReaderError(
                f"contract has {len(contract.weights)} weights but the label length is {k}"
            )
        columns = _normalise_band(band, self.height)
        n_columns = len(columns)
        if not 1 <= n_columns <= MAX_BAND_COLUMNS:
            raise LabelReaderError(f"band must have 1..{MAX_BAND_COLUMNS} columns")
        modulus = contract.modulus

        # layers[i] maps (position, residue) -> up to two distinct prefix
        # strings, each carrying its last witness edge.  Two is an exact
        # bound, not a search cap: see the module docstring for the argument.
        layers: list[dict[tuple[int, int], _LayerEntry]] = [
            {(0, 0): _LayerEntry((_Prefix((), None),))}
        ]
        for i in range(k):
            current: dict[tuple[int, int], _LayerEntry] = {}
            weight = contract.weights[i]
            for (position, residue), entry in layers[i].items():
                for prefix in entry.prefixes:
                    for digit, (width, variants) in self._digits.items():
                        start = position
                        end = position + width
                        if end > n_columns:
                            continue
                        for variant in variants:
                            if not self._matches_at(columns, start, variant):
                                continue
                            next_residue = (residue + weight * int(digit)) % modulus
                            key = (end, next_residue)
                            pointer = _Pointer(
                                prev_position=position,
                                prev_residue=residue,
                                digit=digit,
                                start=start,
                                end=end,
                                variant=variant,
                            )
                            new_prefix = _Prefix(prefix.digits + (digit,), pointer)
                            bucket = current.get(key)
                            if bucket is None:
                                current[key] = _LayerEntry((new_prefix,))
                            elif new_prefix.digits not in {
                                existing.digits for existing in bucket.prefixes
                            }:
                                # Different digit string.  A third prefix is
                                # never needed to answer no/one/two.
                                if len(bucket.prefixes) < _STATE_CAP:
                                    current[key] = _LayerEntry(
                                        bucket.prefixes + (new_prefix,)
                                    )
                            # Identical prefix from another cut/sheet/threshold
                            # collapses onto the first witness.
                            break  # variants are alternative sources, not strings
            layers.append(current)

        final = layers[k]
        winners = [
            entry
            for (position, residue), entry in final.items()
            if position == n_columns and residue in contract.allowed_remainders
        ]

        witnesses: list[LabelWitness] = []
        seen_strings: set[str] = set()
        for entry in winners:
            for prefix in entry.prefixes:
                if prefix.pointer is None:
                    continue
                witness = self._materialise(layers, k, prefix.pointer, contract)
                if witness.digits in seen_strings:
                    continue
                seen_strings.add(witness.digits)
                witnesses.append(witness)
                if len(witnesses) == _STATE_CAP:
                    break
            if len(witnesses) == _STATE_CAP:
                break

        if not witnesses:
            status = STATUS_NO_SOLUTION
        elif len(witnesses) == 1:
            status = STATUS_UNIQUE
        else:
            status = STATUS_AMBIGUOUS
        return ReadResult(
            status=status,
            k=k,
            n_columns=n_columns,
            contract_name=contract.name,
            witnesses=tuple(witnesses),
        )

    def _materialise(
        self,
        layers: Sequence[Mapping[tuple[int, int], "_LayerEntry"]],
        k: int,
        last_pointer: _Pointer,
        contract: LabelContract,
    ) -> LabelWitness:
        """Walk predecessor edges back to the depth-0 state and build one witness."""

        characters: list[CharacterWitness] = []
        checksum = 0
        pointer = last_pointer
        for i in range(k - 1, -1, -1):
            variant = pointer.variant
            origin = variant.origins[0]
            characters.append(
                CharacterWitness(
                    position=i,
                    digit=pointer.digit,
                    column_start=pointer.start,
                    column_end=pointer.end,
                    columns=variant.columns,
                    variant_id=variant.variant_id,
                    origin=origin,
                )
            )
            checksum += contract.weights[i] * int(pointer.digit)
            if i == 0:
                if pointer.prev_position != 0 or pointer.prev_residue != 0:
                    raise LabelReaderError("internal error: DP chain does not start at column 0")
                break
            # The edge stored at depth i points at a state of depth i; that
            # state's own incoming edge lives in its layer-i entry.
            prev_entry = layers[i][(pointer.prev_position, pointer.prev_residue)]
            previous = prev_entry.prefixes[0].pointer
            if previous is None:
                raise LabelReaderError("internal error: dangling DP predecessor")
            pointer = previous
        characters.reverse()
        digits = "".join(char.digit for char in characters)
        return LabelWitness(
            digits=digits,
            checksum=checksum % contract.modulus,
            characters=tuple(characters),
        )
