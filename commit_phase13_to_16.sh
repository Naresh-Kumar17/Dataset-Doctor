#!/bin/bash
# Shell script to create scheduled commits for Phases 13, 14, 15, and 16
# Dates: Strictly between Nov 06, 2026 and Nov 10, 2026
# Total Commits: 10 commits (between 7 and 13)

set -e

make_commit() {
    local date="$1"
    local msg="$2"
    shift 2
    local files=("$@")
    for f in "${files[@]}"; do
        git add "$f"
    done
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
    echo "Committed: $msg ($date)"
}

# 1. Phase 13: Planted dataset and generator
make_commit "2026-11-06T19:30:00+05:00" \
    "feat(phase13): add planted multi-issue pipeline dataset and generator" \
    "data/planted/pipeline.csv" "data/planted/pipeline_truth.json" "tools/make_planted.py"

# 2. Phase 13: Pipeline engine and tests
make_commit "2026-11-06T21:45:00+05:00" \
    "feat(phase13): implement causal pipeline ordering, validator, and before/after comparison" \
    "doctor/pipeline.py" "tests/test_pipeline_ordering.py"

# 3. Phase 14: Quality Assessment core
make_commit "2026-11-07T10:15:00+05:00" \
    "feat(phase14): implement unified quality assessor and 6-domain risk matrix" \
    "doctor/quality.py"

# 4. Phase 14: Quality Assessment tests & blockers
make_commit "2026-11-07T16:30:00+05:00" \
    "feat(phase14): add model training blocker checks, override mechanism, and unit tests" \
    "tests/test_quality_assessment.py"

# 5. Phase 15: Multi-Model Experimentation core
make_commit "2026-11-08T11:00:00+05:00" \
    "feat(phase15): implement multi-model experiment engine across 4 model architectures" \
    "doctor/experiment.py"

# 6. Phase 15: Multi-Model tests & guards
make_commit "2026-11-08T17:20:00+05:00" \
    "feat(phase15): add multi-metric matrix, overfitting detection, and accuracy illusion guard" \
    "tests/test_multi_model.py"

# 7. Phase 16: Model Error Analysis core
make_commit "2026-11-09T10:30:00+05:00" \
    "feat(phase16): implement model error analyzer and confusion matrix diagnostics" \
    "doctor/diagnostics.py"

# 8. Phase 16: Error Analysis tests & slicing
make_commit "2026-11-09T16:45:00+05:00" \
    "feat(phase16): add subgroup slicing, blind spot detection, and error diagnostics tests" \
    "tests/test_error_analysis.py"

# 9. Web UI & Routes integration
make_commit "2026-11-10T11:15:00+05:00" \
    "feat(web): integrate interactive neo-brutalist workbenches and routes for phases 13 to 16" \
    "app/routes/issues.py" "app/templates/module.html" "tests/test_web_routes.py"

# 10. Registry, Benchmarks, Documentation & Commit Scripts
make_commit "2026-11-10T16:00:00+05:00" \
    "docs: register gates 13-16, update gate log, progress tracker, and master phase benchmarks" \
    "doctor/registry.py" "doctor/bench.py" "docs/gate_log.md" "progress.txt" "README.md" "tests/test_registry.py" "commit_phase13_to_16.ps1" "commit_phase13_to_16.sh"

echo "All 10 scheduled commits completed successfully!"
