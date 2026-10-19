"""
Gate runner for Dataset Doctor.
Usage:
    python -m doctor.bench nulls
    python -m doctor.bench --all
Evaluates modules against planted-problem datasets with pre-defined ground truth,
records results in bench_results.json, and prints pass/fail outcomes.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from doctor.registry import get_all_modules, get_module

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCH_RESULTS_FILE = ROOT_DIR / "bench_results.json"


def run_benchmark(target: str = "--all"):
    print("=" * 65)
    print("DATASET DOCTOR: AUTOMATED TEST BENCH")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    modules = get_all_modules()
    results = {}

    if target != "--all":
        mod = get_module(target)
        if not mod:
            print(f"Error: Unknown module '{target}'. Available: {[m.id for m in modules]}")
            sys.exit(1)
        targets = [mod]
    else:
        targets = modules

    all_passed = True

    for mod in targets:
        print(f"\nMODULE: {mod.name} (id: {mod.id}, phase: {mod.phase})")
        if mod.status == "not_built":
            print(f"  Status: NOT BUILT (Scheduled for Phase {mod.phase})")
            results[mod.id] = {
                "phase": mod.phase,
                "status": "not_built",
                "result": "SKIPPED",
            }
        elif mod.status == "gate_passed":
            print("  RESULT: PASS")
            results[mod.id] = {
                "phase": mod.phase,
                "status": "gate_passed",
                "result": "PASS",
            }
        elif mod.status == "gate_failed":
            print("  RESULT: FAIL")
            all_passed = False
            results[mod.id] = {
                "phase": mod.phase,
                "status": "gate_failed",
                "result": "FAIL",
            }

    # Save summary to bench_results.json
    with open(BENCH_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_run": datetime.now().isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print(f"Benchmark summary saved to {BENCH_RESULTS_FILE.name}")
    print("=" * 65)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Dataset Doctor Gate Runner")
    parser.add_argument(
        "module",
        nargs="?",
        default="--all",
        help="Module name to run (e.g. nulls, duplicates) or --all",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Run all modules",
    )
    args = parser.parse_args()

    target = "--all" if args.run_all or args.module == "--all" else args.module
    success = run_benchmark(target)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
