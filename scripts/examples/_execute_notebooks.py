"""Execute notebooks/*.ipynb and report pass/fail."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "results" / "notebooks" / "executed"
LOG = ROOT / "notebook_execute.log"

KERNEL = "brb-py312"
TIMEOUT = 3600  # seconds per cell

NOTEBOOKS = [
    "00_steelmpf_parameters.ipynb",
    "01_single_specimen.ipynb",
    "02_individual.ipynb",
    "03_generalized.ipynb",
    "04_generalized_l2_l1.ipynb",
]


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def run_one(name: str) -> tuple[bool, str]:
    path = NB_DIR / name
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(ROOT)}},
    )
    t0 = time.time()
    try:
        client.execute()
        # Save outputs into the notebook itself (and a results/ backup).
        nbformat.write(nb, path)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, OUT_DIR / name)
        return True, f"OK in {time.time() - t0:.1f}s  (saved outputs in notebooks/{name})"
    except CellExecutionError as e:
        msg = _strip_ansi(str(e))
        return False, f"FAIL after {time.time() - t0:.1f}s\n{msg[-6000:]}"
    except Exception as e:
        return False, f"FAIL after {time.time() - t0:.1f}s: {type(e).__name__}: {e}"


def main(names: list[str] | None = None) -> int:
    targets = names or NOTEBOOKS
    lines: list[str] = []
    rc = 0
    for name in targets:
        header = f"\n======== EXECUTE notebooks/{name} ========\n"
        print(header, flush=True)
        lines.append(header)
        ok, detail = run_one(name)
        print(detail, flush=True)
        lines.append(detail + "\n")
        if not ok:
            rc = 1
            break
    LOG.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {LOG}", flush=True)
    return rc


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.endswith(".ipynb")]
    raise SystemExit(main(args or None))
