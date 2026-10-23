"""
Dataset Doctor: Missing Values Fix Advisor.
Rules (Phase 4):
- Evaluates remediation options strictly on TRAIN data only.
- Combines rule-based statistical shortlisting with honest cross-validated measurement.
- If score difference between top 2 options is within 1 std dev, confidence is 'close call'
  and the simpler option is chosen according to the established simplicity hierarchy:
  leave < median/mode < mean < constant/missing_label < add_indicator < drop_rows < drop_column.
- If dataset < 500 rows, confidence is 'not enough data' and rule-based choice is recommended.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from doctor.base import Advisor, Decision, DetectionContext, Issue, Option, Recommendation
from doctor.issues.nulls.fix import NullFixer

SIMPLICITY_ORDER = {
    "leave": 0,
    "median": 1,
    "mode": 1,
    "mean": 2,
    "constant": 3,
    "missing_label": 3,
    "ffill_interpolate": 4,
    "add_indicator": 5,
    "drop_rows": 6,
    "drop_column": 7,
}


class NullAdvisor(Advisor):
    """
    Evaluates, ranks, and recommends fix strategies for missing value issues.
    """
    id: str = "nulls_advisor"

    def advise(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str] = None,
        ctx: Optional[DetectionContext] = None,
    ) -> Recommendation:
        """
        Generates evidence-backed recommendations for a single missing value issue.
        """
        col = issue.column
        evidence = issue.evidence or {}
        task_type = ctx.task_type if ctx else ("classification" if target else None)
        random_seed = ctx.random_seed if ctx else 42

        # 1. Rule-Based Shortlisting
        shortlist, rule_hits, primary_reason = self._generate_shortlist(issue, train_df, target)

        # Ensure 'leave' is always included as a baseline comparison option
        options_to_test = list(shortlist)
        if "leave" not in options_to_test:
            options_to_test.append("leave")

        # 2. Measurement via Cross-Validation on train_df
        can_measure = (
            target is not None
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
                task_type=task_type,
                random_seed=random_seed,
            )
        else:
            # Fallback when CV cannot be executed (e.g. target missing, EDA mode)
            n_rows = len(train_df)
            for opt_id in options_to_test:
                n_kept = n_rows
                if opt_id == "drop_rows" and col in train_df.columns:
                    n_kept = int(train_df[col].dropna().shape[0])
                measured_options.append(
                    Option(
                        id=opt_id,
                        label=self._get_option_label(opt_id),
                        explanation=self._get_option_explanation(opt_id),
                        measured={"score_mean": 0.0, "score_std": 0.0, "n_rows_kept": n_kept, "pct_rows_kept": round(n_kept / max(1, n_rows) * 100, 1)},
                    )
                )

        # Target relationship from detection evidence
        target_pattern = evidence.get("target_pattern", {})
        is_target_linked = bool(target_pattern.get("related", False))

        # 3. Decision Selection & Confidence Scoring
        best_opt, final_reason, confidence = self._rank_and_select(
            shortlist=shortlist,
            measured_options=measured_options,
            train_size=len(train_df),
            primary_reason=primary_reason,
            can_measure=can_measure,
            is_target_linked=is_target_linked,
        )

        return Recommendation(
            issue=issue,
            recommended=best_opt,
            reason=final_reason,
            rule_hits=rule_hits,
            options=measured_options,
            confidence=confidence,
        )

    def _generate_shortlist(
        self,
        issue: Issue,
        df: pd.DataFrame,
        target: Optional[str],
    ) -> Tuple[List[str], List[str], str]:
        """
        Applies expert rules to determine shortlisted fix methods.
        """
        col = issue.column
        evidence = issue.evidence or {}
        total_rows = len(df)
        shortlist: List[str] = []
        rules: List[str] = []
        primary_reason = ""

        # Case 1: Target column itself is missing
        if col == target or evidence.get("is_target", False):
            rules.append("Rule: Target column missingness must be dropped")
            return ["drop_rows"], rules, "Cannot train a supervised model on rows without target values."

        if not col or col not in df.columns:
            return ["leave"], ["Rule: Column missing"], "Column not found in dataset."

        series = df[col]
        missing_count = series.isna().sum()
        missing_pct = missing_count / max(1, total_rows)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # Calculate skewness if numeric
        skewness = 0.0
        if is_numeric:
            clean_s = series.dropna()
            if len(clean_s) >= 3:
                skewness = float(clean_s.skew())

        # Target relationship from detection evidence
        target_pattern = evidence.get("target_pattern", {})
        is_target_linked = target_pattern.get("related", False)

        # Case 2: Extreme missingness (> 60%)
        if missing_pct > 0.60:
            rules.append(f"Rule: High missingness ({missing_pct * 100:.1f}% > 60%)")
            shortlist = ["drop_column", "add_indicator"]
            primary_reason = f"Column is {missing_pct * 100:.1f}% missing; too little real data remains to impute honestly."
            return shortlist, rules, primary_reason

        # Case 3: Missingness is statistically linked to the target
        if is_target_linked:
            rules.append("Rule: Missingness linked to target (carries predictive signal)")
            shortlist = ["add_indicator"]
            if is_numeric:
                shortlist.append("median" if abs(skewness) > 1.0 else "mean")
            else:
                shortlist.extend(["mode", "missing_label"])
            # Never drop rows when missingness has target signal
            primary_reason = "Missingness is correlated with the target; the fact that it is missing carries signal, and dropping rows would bias training."
            return shortlist, rules, primary_reason

        # Case 4: Numeric distributions
        if is_numeric:
            if abs(skewness) > 1.0:
                rules.append(f"Rule: Numeric skewed distribution (skew = {skewness:.2f})")
                shortlist = ["median", "mean"]
                if missing_pct < 0.05 and total_rows >= 500:
                    shortlist.append("drop_rows")
                primary_reason = f"Numeric distribution is skewed (skewness = {skewness:.2f}); median is preferred as mean gets distorted by outliers."
            else:
                rules.append(f"Rule: Numeric symmetric distribution (skew = {skewness:.2f})")
                shortlist = ["mean", "median"]
                if missing_pct < 0.05 and total_rows >= 500:
                    shortlist.append("drop_rows")
                primary_reason = f"Numeric distribution is roughly symmetric (skewness = {skewness:.2f}); mean and median yield comparable results."

        # Case 5: Categorical or Text features
        else:
            rules.append("Rule: Categorical feature")
            shortlist = ["mode", "missing_label"]
            if missing_pct < 0.05 and total_rows >= 500:
                shortlist.append("drop_rows")
            primary_reason = "Categorical feature; mode preserves existing categories while missing_label creates an explicit MISSING category."

        # Guard: Check if dropping rows would lose > 20% of data
        if missing_pct > 0.20 and "drop_rows" in shortlist:
            rules.append("Guard: Dropping rows would lose > 20% of data (removed drop_rows)")
            shortlist = [opt for opt in shortlist if opt != "drop_rows"]

        return shortlist, rules, primary_reason

    def _measure_options(
        self,
        options: List[str],
        col: str,
        train_df: pd.DataFrame,
        target: str,
        task_type: Optional[str],
        random_seed: int,
    ) -> List[Option]:
        """
        Runs CV on train_df for each candidate option using fixed-model evaluation.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import StratifiedKFold, KFold
        from sklearn.metrics import f1_score, r2_score

        fixer = NullFixer()
        results: List[Option] = []
        is_classification = (task_type == "classification")
        y_train_raw = train_df[target].values
        n_total_rows = len(train_df)

        for opt_id in options:
            decision = Decision(
                issue_id="null_eval",
                column=col,
                option_id=opt_id,
                params={},
                decided_by="user",
            )

            # Measure rows retained if this option was fitted on train
            fitted_global = fixer.fit(train_df, decision)
            transformed_global = fixer.transform(train_df, fitted_global, is_test=False)
            n_rows_kept = len(transformed_global)

            # Run 5-fold CV on train_df
            cv_scores: List[float] = []
            try:
                if is_classification:
                    # Stratified K-Fold
                    counts = np.bincount(y_train_raw.astype(int)) if np.issubdtype(y_train_raw.dtype, np.integer) else pd.Series(y_train_raw).value_counts().values
                    min_samples = int(np.min(counts)) if len(counts) > 0 else 5
                    n_splits = max(2, min(5, min_samples))
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
                    splits = list(skf.split(train_df, y_train_raw))
                else:
                    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
                    splits = list(kf.split(train_df, y_train_raw))

                for fold_tr_idx, fold_val_idx in splits:
                    fold_train = train_df.iloc[fold_tr_idx].copy()
                    fold_val = train_df.iloc[fold_val_idx].copy()

                    # Fit fix strictly on fold_train, transform both fold_train and fold_val
                    fold_fitted = fixer.fit(fold_train, decision)
                    fold_train_fixed = fixer.transform(fold_train, fold_fitted, is_test=False)
                    # Validation fold acts as test (never drops rows)
                    fold_val_fixed = fixer.transform(fold_val, fold_fitted, is_test=True)

                    # Separate X and y
                    if col == target or target not in fold_train_fixed.columns:
                        continue

                    X_tr = fold_train_fixed.drop(columns=[target])
                    y_tr = fold_train_fixed[target]
                    X_val = fold_val_fixed.drop(columns=[target])
                    y_val = fold_val_fixed[target]

                    # Numeric/categorical simple encode for model
                    X_tr_enc, X_val_enc = self._fast_encode(X_tr, X_val)

                    if is_classification:
                        clf = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=random_seed, n_jobs=-1)
                        clf.fit(X_tr_enc, y_tr)
                        preds = clf.predict(X_val_enc)
                        score = float(f1_score(y_val, preds, average="macro", zero_division=0))
                    else:
                        reg = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=random_seed, n_jobs=-1)
                        reg.fit(X_tr_enc, y_tr)
                        preds = reg.predict(X_val_enc)
                        score = float(r2_score(y_val, preds))

                    cv_scores.append(score)

            except Exception:
                pass

            mean_score = round(float(np.mean(cv_scores)), 4) if cv_scores else 0.0
            std_score = round(float(np.std(cv_scores)), 4) if cv_scores else 0.0

            results.append(
                Option(
                    id=opt_id,
                    label=self._get_option_label(opt_id),
                    explanation=self._get_option_explanation(opt_id),
                    measured={
                        "score_mean": mean_score,
                        "score_std": std_score,
                        "n_rows_kept": n_rows_kept,
                        "pct_rows_kept": round(n_rows_kept / max(1, n_total_rows) * 100, 1),
                    },
                )
            )

        return results

    def _rank_and_select(
        self,
        shortlist: List[str],
        measured_options: List[Option],
        train_size: int,
        primary_reason: str,
        can_measure: bool,
        is_target_linked: bool = False,
    ) -> Tuple[str, str, Literal["clear", "close call", "not enough data"]]:
        """
        Ranks options and determines recommendation confidence.
        """
        if not measured_options:
            fallback = shortlist[0] if shortlist else "leave"
            return fallback, primary_reason, "not enough data"

        # Rule 1: Dataset under 500 rows -> "not enough data"
        if train_size < 500 or not can_measure:
            preferred = shortlist[0] if shortlist else measured_options[0].id
            reason = f"{primary_reason} (Small sample: {train_size} rows < 500; prioritizing rule-based choice)."
            return preferred, reason, "not enough data"

        # Filter to shortlisted options for primary ranking, while keeping measured stats
        valid_opts = [o for o in measured_options if o.id in shortlist]
        if not valid_opts:
            valid_opts = measured_options

        # Sort by mean score descending
        sorted_opts = sorted(valid_opts, key=lambda x: x.measured.get("score_mean", 0.0), reverse=True)

        best = sorted_opts[0]
        if len(sorted_opts) == 1:
            return best.id, f"{primary_reason} (Measured CV score: {best.measured.get('score_mean'):.4f})", "clear"

        second = sorted_opts[1]
        best_score = best.measured.get("score_mean", 0.0)
        best_std = best.measured.get("score_std", 0.0)
        second_score = second.measured.get("score_mean", 0.0)

        # Rule 2: If missingness is statistically linked to target, add_indicator preserves vital signal
        if is_target_linked and "add_indicator" in [o.id for o in valid_opts]:
            add_ind_opt = next(o for o in valid_opts if o.id == "add_indicator")
            ind_score = add_ind_opt.measured.get("score_mean", 0.0)
            if abs(best_score - ind_score) <= max(best_std, 0.015):
                reason = f"{primary_reason} (add_indicator preserves predictive signal with CV score: {ind_score:.4f})."
                return "add_indicator", reason, "clear"

        # Check if score difference is within standard deviation (noise)
        diff = abs(best_score - second_score)
        if diff <= best_std or diff < 0.005:
            # Close call: pick simpler option
            simpler = best if SIMPLICITY_ORDER.get(best.id, 99) <= SIMPLICITY_ORDER.get(second.id, 99) else second
            reason = (
                f"{primary_reason} Note: Performance difference ({diff:.4f}) between '{best.id}' and "
                f"'{second.id}' is within CV standard deviation (±{best_std:.4f}); selecting simpler option '{simpler.id}'."
            )
            return simpler.id, reason, "close call"

        reason = f"{primary_reason} (Highest measured CV score: {best_score:.4f} vs {second.id} {second_score:.4f})."
        return best.id, reason, "clear"

    def _fast_encode(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simple fast encoder for CV measurement."""
        X_tr = X_train.copy()
        X_v = X_val.copy()

        num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
        cat_cols = [c for c in X_tr.columns if c not in num_cols]

        for col in num_cols:
            med = float(X_tr[col].median()) if not X_tr[col].dropna().empty else 0.0
            X_tr[col] = X_tr[col].fillna(med)
            X_v[col] = X_v[col].fillna(med)

        for col in cat_cols:
            X_tr[col] = X_tr[col].fillna("__MISSING__").astype(str)
            X_v[col] = X_v[col].fillna("__MISSING__").astype(str)

        if cat_cols:
            X_tr_enc = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
            X_v_enc = pd.get_dummies(X_v, columns=cat_cols, drop_first=True)
            X_v_enc = X_v_enc.reindex(columns=X_tr_enc.columns, fill_value=0)
            return X_tr_enc, X_v_enc

        return X_tr, X_v

    def _get_option_label(self, opt_id: str) -> str:
        labels = {
            "drop_rows": "Drop Rows (Train Only)",
            "drop_column": "Drop Column",
            "mean": "Fill with Mean",
            "median": "Fill with Median",
            "mode": "Fill with Mode",
            "constant": "Fill with Constant",
            "missing_label": "Category 'MISSING'",
            "ffill_interpolate": "Forward-Fill / Interpolate",
            "add_indicator": "Add Missing Indicator",
            "leave": "Leave as-is (NaN)",
        }
        return labels.get(opt_id, opt_id.replace("_", " ").title())

    def _get_option_explanation(self, opt_id: str) -> str:
        exps = {
            "drop_rows": "Removes rows with nulls in train set only. Test rows are untouched.",
            "drop_column": "Removes this column completely from training and test sets.",
            "mean": "Imputes missing values with training mean.",
            "median": "Imputes missing values with training median (robust to outliers).",
            "mode": "Imputes missing values with the most frequent training category.",
            "constant": "Imputes missing values with a designated fixed value.",
            "missing_label": "Encodes missingness as a new explicit category 'MISSING'.",
            "ffill_interpolate": "Fills missing values using time-ordered forward fill or linear interpolation.",
            "add_indicator": "Adds a binary was_missing column and imputes the feature.",
            "leave": "Preserves missing values without modification.",
        }
        return exps.get(opt_id, "")
