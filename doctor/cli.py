"""
Dataset Doctor: Terminal Command-Line Interface (Phase 18).
Commands:
- doctor profile <csv_file> [--target <col>] [--json]
- doctor audit <csv_file> --target <col> [--out <report.json>] [--strict]
- doctor evaluate <csv_file> --target <col> [--test-size <float>] [--fast]
- doctor check-leaks <csv_file> --target <col> [--strict]
- doctor bench [--all | <module>]
- doctor version
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from doctor import __version__
from doctor.api import DatasetDoctor
from doctor.loader import load_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doctor",
        description="Dataset Doctor: Honest, leak-free tabular data preparation & diagnostic CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. profile
    profile_p = subparsers.add_parser("profile", help="Generate dataset profile")
    profile_p.add_argument("file", help="Path to CSV file")
    profile_p.add_argument("--target", default=None, help="Target feature name")
    profile_p.add_argument("--json", action="store_true", help="Output profile in JSON format")

    # 2. audit
    audit_p = subparsers.add_parser("audit", help="Audit dataset quality across all 6 diagnostic domains")
    audit_p.add_argument("file", help="Path to CSV file")
    audit_p.add_argument("--target", required=True, help="Target feature name")
    audit_p.add_argument("--out", default=None, help="Optional output JSON path for full report")
    audit_p.add_argument("--strict", action="store_true", help="Exit with code 1 if critical blockers detected")

    # 3. evaluate
    eval_p = subparsers.add_parser("evaluate", help="Train baseline and evaluate on locked test partition")
    eval_p.add_argument("file", help="Path to CSV file")
    eval_p.add_argument("--target", required=True, help="Target feature name")
    eval_p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default 0.2)")
    eval_p.add_argument("--fast", action="store_true", help="Fast evaluation mode")

    # 4. check-leaks
    leak_p = subparsers.add_parser("check-leaks", help="Check for direct and proxy data leakage")
    leak_p.add_argument("file", help="Path to CSV file")
    leak_p.add_argument("--target", required=True, help="Target feature name")
    leak_p.add_argument("--strict", action="store_true", help="Exit with code 1 if high-severity leaks found")

    # 5. bench
    bench_p = subparsers.add_parser("bench", help="Run automated gate benchmark checks")
    bench_p.add_argument("module", nargs="?", default="--all", help="Module id or --all")
    bench_p.add_argument("--all", dest="run_all", action="store_true", help="Run all modules")

    # 6. consult
    consult_p = subparsers.add_parser("consult", help="Obtain AI Doctor clinical consultation and discharge summary")
    consult_p.add_argument("file", help="Path to CSV file")
    consult_p.add_argument("--target", default=None, help="Target feature name")
    consult_p.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai", "offline"], help="AI provider (default: auto)")
    consult_p.add_argument("--key", default=None, help="Optional API key override")
    consult_p.add_argument("--out", default=None, help="Optional output path to save markdown discharge summary")
    consult_p.add_argument("--json", action="store_true", help="Output raw consultation report in JSON format")

    # 7. version
    subparsers.add_parser("version", help="Print Dataset Doctor version")

    # 7. mcp
    subparsers.add_parser("mcp", help="Run Model Context Protocol (MCP) JSON-RPC stdio server")

    return parser


def cmd_profile(args: argparse.Namespace) -> int:
    doc = DatasetDoctor.from_csv(args.file, target=args.target)
    prof = doc.profile()

    if getattr(args, "json", False):
        data = {
            "n_rows": prof.n_rows,
            "n_cols": prof.n_cols,
            "task_type": prof.task_type,
            "target_column": prof.target_column,
            "total_null_cells": prof.total_null_cells,
            "null_cell_pct": prof.null_cell_pct,
            "memory_usage_mb": prof.memory_usage_mb,
            "columns": [
                {
                    "name": c.name,
                    "inferred_type": c.inferred_type,
                    "null_count": c.null_count,
                    "null_pct": c.null_pct,
                    "unique_count": c.unique_count,
                }
                for c in prof.columns
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print("=" * 60)
        print(f"DATASET DOCTOR: PROFILE ({Path(args.file).name})")
        print("=" * 60)
        print(f"Shape:       {prof.n_rows} rows x {prof.n_cols} columns")
        print(f"Target:      {prof.target_column or 'None specified'}")
        print(f"Task Type:   {prof.task_type or 'unsupervised'}")
        print(f"Missing:     {prof.total_null_cells} cells ({prof.null_cell_pct:.1%})")
        print(f"Memory:      {prof.memory_usage_mb:.2f} MB")
        print("-" * 60)
        print(f"{'COLUMN':<25} {'INFERRED TYPE':<15} {'NULLS':<10} {'UNIQUE':<10}")
        print("-" * 60)
        for col in prof.columns:
            print(f"{col.name:<25} {col.inferred_type:<15} {col.null_count:<10} {col.unique_count:<10}")
        print("=" * 60)

    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    doc = DatasetDoctor.from_csv(args.file, target=args.target)
    assessment = doc.audit()

    print("=" * 65)
    print(f"DATASET DOCTOR: QUALITY AUDIT ({Path(args.file).name})")
    print("=" * 65)
    print(f"Overall Quality Score: {assessment.quality_score} / 100")
    print(f"Training Readiness:    {assessment.readiness}")
    print(f"Can Train Model:       {'YES' if assessment.can_train else 'NO (Blocked by Critical Risks)'}")
    print("-" * 65)
    print("DOMAIN RISK MATRIX:")
    for ds in assessment.domain_summaries.values():
        status_symbol = "✓" if ds.unresolved_issues == 0 else "✗"
        print(f"  [{status_symbol}] {ds.domain_name.upper():<20} Unresolved: {ds.unresolved_issues:<3}")
        for find in ds.findings[:2]:
            print(f"      • {find}")

    if assessment.blocker_reasons:
        print("\nCRITICAL TRAINING BLOCKERS:")
        for b in assessment.blocker_reasons:
            print(f"  [!] {b}")

    print("=" * 65)

    if args.out:
        out_path = Path(args.out)
        out_data = {
            "quality_score": assessment.quality_score,
            "readiness": assessment.readiness,
            "can_train": assessment.can_train,
            "blocker_reasons": assessment.blocker_reasons,
            "domains": [
                {
                    "domain_id": ds.domain_id,
                    "domain_name": ds.domain_name,
                    "total_issues": ds.total_issues,
                    "unresolved_issues": ds.unresolved_issues,
                    "resolved_issues": ds.resolved_issues,
                    "findings": ds.findings,
                }
                for ds in assessment.domain_summaries.values()
            ],
            "total_issues": len(assessment.raw_issues),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Audit report saved to: {out_path.name}")

    if args.strict and not assessment.can_train:
        return 1

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    doc = DatasetDoctor.from_csv(args.file, target=args.target)
    doc.split(test_size=args.test_size)
    eval_res = doc.evaluate(fast_mode=args.fast)

    print("=" * 65)
    print(f"DATASET DOCTOR: MODEL EVALUATION ({Path(args.file).name})")
    print("=" * 65)
    print(f"Target:          {args.target}")
    print(f"Primary Metric:  {eval_res.primary_metric.upper()}")
    print("-" * 65)
    print("TRAIN 5-FOLD CV SCORES (Mean ± Std):")
    for metric, mean_val in eval_res.train_cv_mean.items():
        std_val = eval_res.train_cv_std.get(metric, 0.0)
        print(f"  {metric.upper():<16}: {mean_val:.4f} ± {std_val:.4f}")

    print("-" * 65)
    print("HONEST TEST SET PERFORMANCE (Untouched Partition):")
    for metric, val in eval_res.test_scores.items():
        print(f"  {metric.upper():<16}: {val:.4f}")

    if not eval_res.accuracy_reliable:
        print("\n⚠️ WARNING: Class imbalance detected. Accuracy is deceptive; rely on Minority F1 and PR-AUC.")

    print("=" * 65)
    return 0


def cmd_check_leaks(args: argparse.Namespace) -> int:
    doc = DatasetDoctor.from_csv(args.file, target=args.target)
    leak_issues = doc.check_leakage()

    print("=" * 65)
    print(f"DATASET DOCTOR: LEAKAGE SCAN ({Path(args.file).name})")
    print("=" * 65)

    high_leaks = [i for i in leak_issues if i.severity == "high"]
    med_leaks = [i for i in leak_issues if i.severity == "medium"]

    if not leak_issues:
        print("✓ Zero data leakage detected. Features are honest predictors.")
        print("=" * 65)
        return 0

    print(f"Detected {len(leak_issues)} leakage risk(s): {len(high_leaks)} HIGH, {len(med_leaks)} MEDIUM")
    print("-" * 65)
    for iss in leak_issues:
        sev_tag = f"[{iss.severity.upper()}]"
        print(f"  {sev_tag:<8} Column: {iss.column}")
        print(f"           Title:  {iss.title}")
        print(f"           Reason: {iss.description}")
        if "abs_correlation" in iss.evidence:
            print(f"           Evidence: Pearson r = {iss.evidence['abs_correlation']:.4f}")
        print()

    print("=" * 65)
    if args.strict and high_leaks:
        return 1

    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from doctor.bench import run_benchmark
    target = "--all" if args.run_all or args.module == "--all" else args.module
    success = run_benchmark(target)
    return 0 if success else 1


def cmd_consult(args) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{args.file}' does not exist.", file=sys.stderr)
        return 1
    doc = DatasetDoctor.from_csv(args.file, target=args.target)
    report = doc.ai_consult(provider=args.provider, api_key=args.key)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.discharge_letter_markdown)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.discharge_letter_markdown, encoding="utf-8")
        print(f"\n[OK] Clinical Discharge Summary saved to {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "version":
        print(f"Dataset Doctor CLI version {__version__}")
        return 0
    elif args.command == "profile":
        return cmd_profile(args)
    elif args.command == "audit":
        return cmd_audit(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "check-leaks":
        return cmd_check_leaks(args)
    elif args.command == "bench":
        return cmd_bench(args)
    elif args.command == "consult":
        return cmd_consult(args)
    elif args.command == "mcp":
        from doctor.mcp import DatasetDoctorMCPServer
        server = DatasetDoctorMCPServer()
        server.run_stdio()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

