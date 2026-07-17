#!/usr/bin/env python3
"""Run the GC mark-phase benchmark suite.

Builds the `gc_bench` binary (release) and times the GC mark phase for every
workload in `benchmark_scripts/gc/*.shm`, reporting the minimum per-iteration
time over several trials (the min is the least noise-contaminated estimate for
CPU-bound work).

Usage:
    python3 benchmark_scripts/gc_bench.py [--iterations N] [--trials T]
                                          [--baseline PATH_TO_gc_bench]

`--baseline` points at another `gc_bench` binary to print a before/after delta
per workload -- handy for catching GC-mark regressions. The baseline revision
must also contain this benchmark tooling:

    git worktree add /tmp/base <ref>
    (cd /tmp/base && cargo build --release --bin gc_bench)
    python3 benchmark_scripts/gc_bench.py --baseline /tmp/base/target/release/gc_bench
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKLOAD_DIR = ROOT / "benchmark_scripts" / "gc"


def build(manifest_dir: Path) -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "gc_bench"],
        cwd=manifest_dir,
        check=True,
    )
    return manifest_dir / "target" / "release" / "gc_bench"


def measure(binary: Path, script: Path, iterations: int, trials: int) -> float:
    best = float("inf")
    for _ in range(trials):
        # gc_bench reports timing on stderr; stdout carries the script's output.
        out = subprocess.run(
            [str(binary), str(script), str(iterations)],
            capture_output=True,
            text=True,
            check=True,
        ).stderr
        best = min(best, float(out.split()[0]))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="another gc_bench binary to compare against")
    args = ap.parse_args()

    current = build(ROOT)
    workloads = sorted(WORKLOAD_DIR.glob("*.shm"))
    if not workloads:
        print(f"no workloads found in {WORKLOAD_DIR}", file=sys.stderr)
        return 1

    if args.baseline:
        print(f"{'workload':<14}{'baseline(ms)':>13}{'current(ms)':>13}{'delta':>8}")
        print("-" * 48)
        deltas = []
        for w in workloads:
            base = measure(args.baseline, w, args.iterations, args.trials)
            cur = measure(current, w, args.iterations, args.trials)
            d = (cur - base) / base * 100.0
            deltas.append(d)
            print(f"{w.stem:<14}{base:>13.4f}{cur:>13.4f}{d:>+7.1f}%")
        print("-" * 48)
        print(f"{'mean':<14}{'':>26}{sum(deltas) / len(deltas):>+7.1f}%")
        print(f"{'worst':<14}{'':>26}{max(deltas):>+7.1f}%")
    else:
        print(f"{'workload':<14}{'mark(ms/iter)':>14}")
        print("-" * 28)
        for w in workloads:
            print(f"{w.stem:<14}{measure(current, w, args.iterations, args.trials):>14.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
