"""Small command-line entry point for design editing, publishing and fitting.

Run ``python -m plate_effects.cli demo`` to execute the bundled synthetic
scenarios and write JSON reports under ``results/``.
"""

from __future__ import annotations

import json
import sys

from .scenarios import build_all_scenarios, scenario_to_dict
from .warehouse import clear_caches


def _dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def demo(outdir: str = "results") -> int:
    import os

    os.makedirs(outdir, exist_ok=True)
    clear_caches()
    scenarios = build_all_scenarios()
    summary = {}
    for name, result in scenarios.items():
        summary[name] = scenario_to_dict(result)
        _dump(os.path.join(outdir, f"{name}.json"), summary[name])
    _dump(os.path.join(outdir, "summary.json"), summary)
    for name, payload in summary.items():
        print(f"{name}: {payload['headline']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m plate_effects.cli demo [outdir]")
        return 0
    if argv[0] == "demo":
        return demo(argv[1] if len(argv) > 1 else "results")
    print(f"unknown command {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
