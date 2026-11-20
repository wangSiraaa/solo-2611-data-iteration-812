"""Small command-line entry point for design editing, publishing and fitting.

Run ``python -m plate_effects.cli demo`` to execute the bundled synthetic
scenarios and write JSON reports under ``results/``.  Run
``python -m plate_effects.cli label-demo`` to build glyph fonts from synthetic
correction runs, export them and read unseparated scan bands offline.
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


def label_demo(outdir: str = "results") -> int:
    import os

    from .label_scenarios import build_all_demos

    os.makedirs(outdir, exist_ok=True)
    results = build_all_demos(outdir)
    _dump(os.path.join(outdir, "label_summary.json"), results)
    for name in results:
        print(f"{name}: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage:\n"
            "  python -m plate_effects.cli demo [outdir]\n"
            "  python -m plate_effects.cli label-demo [outdir]"
        )
        return 0
    if argv[0] == "demo":
        return demo(argv[1] if len(argv) > 1 else "results")
    if argv[0] == "label-demo":
        return label_demo(argv[1] if len(argv) > 1 else "results")
    print(f"unknown command {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
