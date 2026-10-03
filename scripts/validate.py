from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    _run(
        [sys.executable, "-m", "pytest", "backend/tests", "-q"],
        "BACKEND TESTS",
    )

    npm_name = "npm.cmd" if os.name == "nt" else "npm"
    npm = shutil.which(npm_name) or shutil.which("npm")
    if npm is None:
        raise SystemExit("npm was not found on PATH; frontend validation cannot run.")

    _run(
        [npm, "--prefix", "frontend", "run", "build"],
        "FRONTEND PRODUCTION BUILD",
    )

    print("\n=== VALIDATION PASSED ===", flush=True)


if __name__ == "__main__":
    main()
