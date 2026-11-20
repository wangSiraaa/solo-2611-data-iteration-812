"""Exact rational helpers and tiny exact linear-algebra kernel.

All model arithmetic in this project is done over ``fractions.Fraction``.
That matters for identifiability: a structural zero in the design matrix must
be recognised as exactly zero, not as a small floating-point singular value.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Iterable, Sequence

Matrix = list[list[Fraction]]
Vector = list[Fraction]


class RationalError(ValueError):
    """Raised when an input cannot be interpreted as an exact rational."""


def parse_rational(value: object, *, field: str = "value") -> Fraction:
    """Parse an exact rational.

    Accepted forms: ``int``, ``Fraction``, ``Decimal``, a string such as
    ``"3/4"`` or ``"-1.25"``, and (for convenience) ``float``.  Floats are
    converted through their shortest decimal spelling, never through their
    binary expansion.
    """

    if isinstance(value, bool):
        raise RationalError(f"{field}: booleans are not rational values")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise RationalError(f"{field}: non-finite decimal")
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RationalError(f"{field}: empty rational string")
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise RationalError(f"{field}: invalid rational {value!r}") from exc
    if isinstance(value, float):
        # Exact value of the shortest decimal representation.
        if value != value or value in (float("inf"), float("-inf")):
            raise RationalError(f"{field}: non-finite float")
        return Fraction(Decimal(str(value)))
    raise RationalError(f"{field}: unsupported rational type {type(value).__name__}")


def fmt_rational(value: Fraction) -> str:
    """Canonical text form: ``"n"`` for integers, otherwise ``"n/d"``."""

    if not isinstance(value, Fraction):
        value = parse_rational(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def fmt_float(value: Fraction) -> float:
    return float(value)


def zero_matrix(rows: int, cols: int) -> Matrix:
    return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]


def identity_matrix(n: int) -> Matrix:
    return [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]


def transpose(matrix: Sequence[Sequence[Fraction]]) -> Matrix:
    if not matrix:
        return []
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def matmul(left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]) -> Matrix:
    if not left:
        return []
    if not right:
        return [[] for _ in left]
    inner = len(right)
    cols = len(right[0])
    if len(left[0]) != inner:
        raise ValueError("matmul dimension mismatch")
    out = zero_matrix(len(left), cols)
    for i, row in enumerate(left):
        for k, aik in enumerate(row):
            if aik == 0:
                continue
            right_row = right[k]
            for j in range(cols):
                bkj = right_row[j]
                if bkj:
                    out[i][j] += aik * bkj
    return out


def matvec(matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]) -> Vector:
    if not matrix:
        return []
    return [sum((a * b for a, b in zip(row, vector)), Fraction(0)) for row in matrix]


def dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    if len(left) != len(right):
        raise ValueError("dot dimension mismatch")
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def rref(matrix: Sequence[Sequence[Fraction]], *, ncols: int | None = None) -> tuple[Matrix, list[int]]:
    """Exact reduced row echelon form.

    Returns ``(rref_matrix, pivot_columns)``.  No tolerance is involved.
    """

    if matrix:
        width = len(matrix[0])
        if any(len(row) != width for row in matrix):
            raise ValueError("ragged matrix")
    else:
        width = 0 if ncols is None else ncols
    work = [list(row) for row in matrix]
    rows = len(work)
    pivots: list[int] = []
    r = 0
    for c in range(width):
        pivot = next((i for i in range(r, rows) if work[i][c] != 0), None)
        if pivot is None:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        pv = work[r][c]
        if pv != 1:
            work[r] = [x / pv for x in work[r]]
        for i in range(rows):
            if i == r:
                continue
            factor = work[i][c]
            if factor:
                work[i] = [x - factor * y for x, y in zip(work[i], work[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return work, pivots


def rank(matrix: Sequence[Sequence[Fraction]], *, ncols: int | None = None) -> int:
    if not matrix:
        return 0
    _, pivots = rref(matrix, ncols=ncols)
    return len(pivots)


def nullspace(matrix: Sequence[Sequence[Fraction]], *, ncols: int | None = None) -> list[Vector]:
    """Exact basis for the right nullspace of ``matrix``.

    Each returned vector has length ``ncols`` (or the matrix width).
    """

    if matrix:
        width = len(matrix[0])
    else:
        width = 0 if ncols is None else ncols
    reduced, pivots = rref(matrix, ncols=width)
    pivot_set = set(pivots)
    free = [c for c in range(width) if c not in pivot_set]
    basis: list[Vector] = []
    for free_col in free:
        vec = [Fraction(0) for _ in range(width)]
        vec[free_col] = Fraction(1)
        for row_index, pivot_col in enumerate(pivots):
            vec[pivot_col] = -reduced[row_index][free_col]
        basis.append(vec)
    return basis


def solve_linear(
    matrix: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
    *,
    ncols: int | None = None,
) -> tuple[Vector, list[Vector], int, bool]:
    """Solve ``matrix @ x = rhs`` exactly.

    Returns ``(particular, nullspace_basis, rank, consistent)``.  Free
    variables are set to zero in the particular solution.
    """

    if matrix:
        width = len(matrix[0])
        if len(rhs) != len(matrix):
            raise ValueError("rhs length mismatch")
        if any(len(row) != width for row in matrix):
            raise ValueError("ragged matrix")
    else:
        width = 0 if ncols is None else ncols
        if rhs:
            raise ValueError("rhs length mismatch")
    if ncols is not None and ncols != width:
        raise ValueError("ncols mismatch")

    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    reduced, pivots = rref(augmented, ncols=width + 1)

    # Inconsistent iff the last column is a pivot column.
    if pivots and pivots[-1] == width:
        return ([Fraction(0) for _ in range(width)], [], len(pivots) - 1, False)

    rank_value = len(pivots)
    pivot_set = set(pivots)
    particular = [Fraction(0) for _ in range(width)]
    for row_index, pivot_col in enumerate(pivots):
        particular[pivot_col] = reduced[row_index][width]

    free = [c for c in range(width) if c not in pivot_set]
    null_basis: list[Vector] = []
    for free_col in free:
        vec = [Fraction(0) for _ in range(width)]
        vec[free_col] = Fraction(1)
        for row_index, pivot_col in enumerate(pivots):
            vec[pivot_col] = -reduced[row_index][free_col]
        null_basis.append(vec)

    return particular, null_basis, rank_value, True


def rref_augmented(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> tuple[Matrix, list[int], int]:
    """RREF of ``[matrix | rhs]``; helper used by verification output."""

    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    reduced, pivots = rref(augmented)
    return reduced, pivots, len(matrix[0]) if matrix else 0


def sum_squares(values: Iterable[Fraction]) -> Fraction:
    return sum((v * v for v in values), Fraction(0))
