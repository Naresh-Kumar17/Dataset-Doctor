"""
Dataset Doctor: Outliers Fix Advisor (Phase 8).
Recommends principled outlier remediations based on statistical properties,
domain validity, target class rarity, and honest 5-fold CV evaluation.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from doctor.base import Advisor, Decision, DetectionContext, Issue, Option, Recommendation
from doctor.issues.outliers.fix import OutlierFixer

OUTLIER_SIMPLICITY = {
    "leave": 0,
    "clip_percentile": 1,
    "clip_iqr": 2,
    "log_transform": 3,
    "set_to_null": 4,
    "drop_rows": 5,
}


class OutlierAdvisor(Advisor):
    """
    Evaluates, ranks, and recommends outlier remediation options on training data only.
    """
    id: str = "outliers_advisor"

    def __init__(self):
        self._fixer = OutlierFixer()

    def advise(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str] = None,
        ctx: Optional[DetectionContext] = None,
        measure: bool = True,
    ) -> Recommendation:
        """
        Generates evidence-backed recommendations for an outlier issue.
        """
        col = issue.column
        ev = issue.evidence or {}
        subtype = ev.get("subtype", "extreme_tail")
        random_seed = ctx.random_seed if ctx else 42

        # 1. Rule-Based Shortlisting
        shortlist, rule_hits, primary_reason = self._generate_shortlist(issue, train_df, target)

        options_to_test = list(shortlist)
        if "leave" not in options_to_test:
            options_to_test.append("leave")

        # 2. Measurement via Cross-Validation on train_df
        can_measure = (
            measure
            and target is not None
            and target in train_df.columns
            and col in train_df.columns
            and col != target
            and len(train_df) >= 20
        )

        measured_options: List[Option] = []
        if can_measure:
            measured_options = self._measure_options(
                options=options_to_test,
                col=col,
                train_df=train_df,
                target=target,
                random_seed=random_seed,
            )
        else:
            measured_options = [
                Option(
                    id=opt,
                    label=self._get_option_label(opt),
                    params={},
                    explanation=self._get_option_explanation(opt),
                    measured=None,
                )
                for opt in options_to_test
            ]

        # 3. Decision Selection: Rule vs Measurement with Noise Tie-Breaker
        if subtype == "impossible_values":
            # Domain impossible values are ALWAYS set_to_null (non-negotiable domain rule)
            recommended = "set_to_null"
            reason = primary_reason
            confidence: Literal["clear", "close call", "not enough data"] = "clear"
        elif can_measure and len(measured_options) >= 2:
            scored = [o for o in measured_options if o.measured is not None and "f1_mean" in o.measured]
            if scored:
                scored.sort(key=lambda o: o.measured["f1_mean"], reverse=True)
                top = scored[0]
                second = scored[1] if len(scored) > 1 else None

                leave_opt = next((o for o in scored if o.id == "leave"), None)
                delta_with_leave = (top.measured["f1_mean"] - leave_opt.measured["f1_mean"]) if leave_opt else 0.0
                std_noise = top.measured.get("f1_std", 0.015)

                if second and abs(top.measured["f1_mean"] - second.measured["f1_mean"]) <= std_noise:
                    confidence = "close call"
                    # In a close call, prefer simpler option: leave > clip_percentile > log_transform
                    candidates = [top, second]
                    if leave_opt and abs(top.measured["f1_mean"] - leave_opt.measured["f1_mean"]) <= std_noise:
                        candidates.append(leave_opt)
                    candidates.sort(key=lambda o: OUTLIER_SIMPLICITY.get(o.id, 99))
                    chosen = candidates[0]
                    recommended = chosen.id
                    if chosen.id == "leave":
                        reason = (
                            f"Cross-validation indicates score changes are within noise (Δ={delta_with_leave:.4f} <= σ={std_noise:.4f}). "
                            "Following the simplicity principle, leaving values untouched is recommended."
                        )
                    else:
                        reason = (
                            f"Score difference is within statistical noise (Δ <= {std_noise:.4f}). "
                            f"Recommended simpler tie-breaker: '{chosen.id}'."
                        )
                else:
                    confidence = "clear"
                    recommended = top.id
                    reason = f"Measured top performance in 5-fold CV: F1 = {top.measured['f1_mean']:.3f} ± {std_noise:.3f}."
            else:
                recommended = shortlist[0]
                reason = primary_reason
                confidence = "clear"
        else:
            recommended = shortlist[0]
            reason = primary_reason
            confidence = "clear" if len(train_df) >= 100 else "not enough data"

        return Recommendation(
            issue=issue,
            recommended=recommended,
            reason=reason,
            rule_hits=rule_hits,
            options=measured_options,
            confidence=confidence,
        )

    def _generate_shortlist(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str],
    ) -> Tuple[List[str], List[str], str]:
        ev = issue.evidence or {}
        subtype = ev.get("subtype", "extreme_tail")
        is_skewed = ev.get("is_skewed", False)
        col = issue.column

        rule_hits: List[str] = []

        if subtype == "impossible_values":
            rule_hits.append("RULE_IMPOSSIBLE_VALUES_SET_TO_NULL")
            shortlist = ["set_to_null", "leave"]
            reason = "Impossible values are invalid data errors; converting to null enables principled imputation."
            return shortlist, rule_hits, reason

        if subtype == "unit_mixup":
            rule_hits.append("RULE_UNIT_MIXUP_CONVERT_OR_CLIP")
            shortlist = ["set_to_null", "clip_percentile", "leave"]
            reason = "Unit errors heavily distort metric distances; converting to null or clipping restores scale."
            return shortlist, rule_hits, reason

        # Extreme tail outliers:
        # Check if target is rare (< 20% positive class)
        target_is_rare = False
        minority_rows_in_outliers = False

        if target and target in train_df.columns:
            target_series = train_df[target].dropna()
            if target_series.nunique() == 2:
                shares = target_series.value_counts(normalize=True)
                min_share = float(shares.min())
                if min_share < 0.20:
                    target_is_rare = True
                    minority_class = shares.idxmin()
                    outlier_indices = ev.get("row_indices", [])
                    if outlier_indices:
                        train_indices = [idx for idx in outlier_indices if idx in train_df.index]
                        min_in_outliers = (train_df.loc[train_indices, target] == minority_class).sum()
                        if min_in_outliers > 0:
                            minority_rows_in_outliers = True
                            rule_hits.append("RULE_RARE_TARGET_PROTECT_MINORITY_EXTREMES")

        if minority_rows_in_outliers or target_is_rare:
            shortlist = ["leave", "clip_percentile", "log_transform"]
            reason = (
                "Extreme values coincide with rare target instances. "
                "Dropping rows would eliminate critical minority cases; leaving untouched is recommended."
            )
            return shortlist, rule_hits, reason

        if is_skewed:
            rule_hits.append("RULE_SKEWED_CONTINUOUS_TRANSFORM")
            shortlist = ["leave", "clip_percentile", "log_transform", "clip_iqr"]
            reason = "Feature exhibits natural skewness; log transform or percentile clipping tames tail without row loss."
            return shortlist, rule_hits, reason

        rule_hits.append("RULE_SYMMETRIC_TAIL_LEAVE_OR_CLIP")
        shortlist = ["leave", "clip_percentile", "clip_iqr", "drop_rows"]
        reason = "Tail extremes can be safely left for tree models, or clipped to train percentiles."
        return shortlist, rule_hits, reason

    def _measure_options(
        self,
        options: List[str],
        col: str,
        train_df: pd.DataFrame,
        target: str,
        random_seed: int = 42,
    ) -> List[Option]:
        results: List[Option] = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        # Baseline evaluation
        for opt in options:
            f1_scores: List[float] = []
            rows_kept_list: List[int] = []

            for train_idx, val_idx in cv.split(train_df, train_df[target]):
                fold_train = train_df.iloc[train_idx].copy()
                fold_val = train_df.iloc[val_idx].copy()

                dec = Decision(
                    issue_id="outliers",
                    column=col,
                    option_id=opt,
                    params={},
                    decided_by="user",
                )
                fitted = self._fixer.fit(fold_train, dec)
                trans_train = self._fixer.transform(fold_train, dec, fitted, is_test=False)
                trans_val = self._fixer.transform(fold_val, dec, fitted, is_test=True)

                rows_kept_list.append(len(trans_train))

                # Train lightweight model on numeric features
                X_tr = trans_train.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore").fillna(0)
                y_tr = trans_train[target]
                X_va = trans_val.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore").fillna(0)
                y_va = trans_val[target]

                # Align columns
                common_cols = [c for c in X_tr.columns if c in X_va.columns]
                if not common_cols or len(X_tr) < 5 or y_tr.nunique() < 2:
                    f1_scores.append(0.5)
                    continue

                clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=random_seed)
                clf.fit(X_tr[common_cols], y_tr)
                preds = clf.predict(X_va[common_cols])
                score = float(f1_score(y_va, preds, zero_division=0))
                f1_scores.append(score)

            mean_score = float(np.mean(f1_scores)) if f1_scores else 0.5
            std_score = float(np.std(f1_scores)) if f1_scores else 0.015

            results.append(
                Option(
                    id=opt,
                    label=self._get_option_label(opt),
                    params={},
                    explanation=self._get_option_explanation(opt),
                    measured={
                        "f1_mean": round(mean_score, 4),
                        "f1_std": round(std_score, 4),
                        "avg_rows_kept": int(np.mean(rows_kept_list)),
                    },
                )
            )

        return results

    def _get_option_label(self, opt_id: str) -> str:
        labels = {
            "leave": "Leave Unchanged",
            "set_to_null": "Convert to NaN (Hand to Null Advisor)",
            "clip_percentile": "Clip to 1st & 99th Percentiles",
            "clip_iqr": "Clip to IQR Bounds (1.5x IQR)",
            "log_transform": "Apply Log1p Transform",
            "drop_rows": "Drop Outlier Rows (Train Only)",
        }
        return labels.get(opt_id, opt_id.replace("_", " ").title())

    def _get_option_explanation(self, opt_id: str) -> str:
        exps = {
            "leave": "Keeps original values intact. Recommended for tree-based models and rare target instances.",
            "set_to_null": "Converts impossible values to null for principled imputation without data snooping.",
            "clip_percentile": "Caps extreme tails to 1st and 99th percentiles learned strictly from train.",
            "clip_iqr": "Caps values exceeding 1.5x IQR to learned training thresholds.",
            "log_transform": "Applies monotonic logarithmic scaling to compress long right-hand tails.",
            "drop_rows": "Drops outlier rows from train set only. Never drops test rows.",
        }
        return exps.get(opt_id, "")
