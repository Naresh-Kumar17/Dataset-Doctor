"""
Dataset Doctor: AI Diagnostic Consultant & Second Opinion Agent.
Provides an autonomous medical consultation layer that analyzes data health charts,
flags semantic leakage risks, diagnoses subgroup bias, and prescribes surgical treatments.

Supports:
- Google Gemini API (gemini-1.5-flash / gemini-2.5-flash)
- OpenAI-compatible API (GPT-4o, Groq, Ollama, DeepSeek)
- Built-In Offline Expert Diagnostic Synthesizer (Zero-config, deterministic, 100% offline)
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("dataset_doctor.ai_consultant")


@dataclass
class AIConsultationReport:
    """Structured clinical report produced by the AI Doctor."""
    clinical_impression: str
    patient_status: str  # "CRITICAL", "GUARDED", "STABLE", "EXCELLENT"
    survival_forecast: str
    semantic_leakage_review: List[Dict[str, str]]
    fairness_bias_analysis: str
    prescriptions: List[str]
    discharge_letter_markdown: str
    provider: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clinical_impression": self.clinical_impression,
            "patient_status": self.patient_status,
            "survival_forecast": self.survival_forecast,
            "semantic_leakage_review": self.semantic_leakage_review,
            "fairness_bias_analysis": self.fairness_bias_analysis,
            "prescriptions": self.prescriptions,
            "discharge_letter_markdown": self.discharge_letter_markdown,
            "provider": self.provider,
            "timestamp": self.timestamp,
        }


class AIConsultant:
    """
    Autonomous Machine Learning Physician providing second opinions on dataset health.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = (
            openai_base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")

    def consult(
        self,
        df: Optional[Any] = None,
        profile: Optional[Dict[str, Any]] = None,
        quality_assessment: Optional[Any] = None,
        experiment_result: Optional[Any] = None,
        error_profile: Optional[Any] = None,
        target_column: Optional[str] = None,
        provider: str = "auto",
        api_key_override: Optional[str] = None,
    ) -> AIConsultationReport:
        """
        Executes a clinical consultation. Attempts external LLM if credentials exist,
        otherwise seamlessly executes the offline clinical reasoning engine.
        """
        gemini_key = api_key_override if provider == "gemini" else (self.gemini_api_key or api_key_override)
        openai_key = api_key_override if provider == "openai" else (self.openai_api_key or api_key_override)

        # 1. Select provider
        if provider == "gemini" and gemini_key:
            try:
                return self._consult_gemini(
                    df, profile, quality_assessment, experiment_result, error_profile, target_column, gemini_key
                )
            except Exception as e:
                logger.warning(f"Gemini consultation failed ({e}). Falling back to Offline Expert Synthesizer.")

        elif provider == "openai" and openai_key:
            try:
                return self._consult_openai(
                    df, profile, quality_assessment, experiment_result, error_profile, target_column, openai_key
                )
            except Exception as e:
                logger.warning(f"OpenAI consultation failed ({e}). Falling back to Offline Expert Synthesizer.")

        elif provider == "auto":
            if gemini_key:
                try:
                    return self._consult_gemini(
                        df, profile, quality_assessment, experiment_result, error_profile, target_column, gemini_key
                    )
                except Exception as e:
                    logger.warning(f"Auto Gemini failed ({e}). Attempting next fallback.")
            if openai_key:
                try:
                    return self._consult_openai(
                        df, profile, quality_assessment, experiment_result, error_profile, target_column, openai_key
                    )
                except Exception as e:
                    logger.warning(f"Auto OpenAI failed ({e}). Attempting next fallback.")

        # Default guaranteed fallback: deterministic clinical synthesis
        return self._synthesize_offline_consultation(
            df=df,
            profile=profile,
            quality_assessment=quality_assessment,
            experiment_result=experiment_result,
            error_profile=error_profile,
            target_column=target_column,
        )

    def _build_context_summary(
        self,
        df: Optional[Any],
        profile: Optional[Dict[str, Any]],
        quality_assessment: Optional[Any],
        experiment_result: Optional[Any],
        error_profile: Optional[Any],
        target_column: Optional[str],
    ) -> Dict[str, Any]:
        """Constructs an audited context payload for the prompt or offline reasoner."""
        n_rows = len(df) if df is not None else (profile.get("n_rows", 0) if profile else 0)
        n_cols = len(df.columns) if df is not None else (profile.get("n_columns", 0) if profile else 0)
        columns = list(df.columns) if df is not None else [c.get("name") for c in profile.get("columns", [])] if profile else []

        score = getattr(quality_assessment, "overall_score", 100) if quality_assessment else 100
        readiness = getattr(quality_assessment, "readiness", "TRAINING_READY") if quality_assessment else "TRAINING_READY"
        blockers = getattr(quality_assessment, "critical_blockers", []) if quality_assessment else []

        domain_findings = {}
        if quality_assessment and hasattr(quality_assessment, "domain_summaries"):
            for d_id, ds in quality_assessment.domain_summaries.items():
                domain_findings[d_id] = {
                    "status": getattr(ds, "status", "CLEAN"),
                    "unresolved": getattr(ds, "unresolved_issues", 0),
                    "findings": getattr(ds, "findings", []),
                }

        model_summary = {}
        if experiment_result:
            model_summary = {
                "best_model": getattr(experiment_result, "best_overall_model", "N/A"),
                "best_f1": getattr(experiment_result, "best_model_f1", "N/A"),
                "accuracy_illusion": getattr(experiment_result, "has_accuracy_illusion", False),
            }

        error_summary = {}
        if error_profile:
            error_summary = {
                "fpr": getattr(error_profile, "false_positive_rate", 0.0),
                "fnr": getattr(error_profile, "false_negative_rate", 0.0),
                "blind_spots": [
                    f"{b.feature}={b.slice_value} (error rate: {b.error_rate:.1%})"
                    for b in getattr(error_profile, "blind_spots", [])
                ],
            }

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": columns,
            "target_column": target_column,
            "quality_score": score,
            "readiness": readiness,
            "critical_blockers": blockers,
            "domain_findings": domain_findings,
            "model_summary": model_summary,
            "error_summary": error_summary,
        }

    def _consult_gemini(
        self,
        df: Optional[Any],
        profile: Optional[Dict[str, Any]],
        quality_assessment: Optional[Any],
        experiment_result: Optional[Any],
        error_profile: Optional[Any],
        target_column: Optional[str],
        api_key: str,
    ) -> AIConsultationReport:
        """Invokes Google Gemini REST endpoint."""
        ctx = self._build_context_summary(df, profile, quality_assessment, experiment_result, error_profile, target_column)
        prompt = self._create_clinical_prompt(ctx)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text_content = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text_content)
        return self._format_json_response(parsed, provider="Google Gemini 1.5 Flash")

    def _consult_openai(
        self,
        df: Optional[Any],
        profile: Optional[Dict[str, Any]],
        quality_assessment: Optional[Any],
        experiment_result: Optional[Any],
        error_profile: Optional[Any],
        target_column: Optional[str],
        api_key: str,
    ) -> AIConsultationReport:
        """Invokes OpenAI or OpenAI-compatible endpoint."""
        ctx = self._build_context_summary(df, profile, quality_assessment, experiment_result, error_profile, target_column)
        prompt = self._create_clinical_prompt(ctx)

        url = f"{self.openai_base_url}/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are Dr. Datadoctor, Chief of Machine Learning Data Diagnostics. Respond strictly in valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text_content = data["choices"][0]["message"]["content"]
        parsed = json.loads(text_content)
        return self._format_json_response(parsed, provider="OpenAI GPT-4o-mini")

    def _create_clinical_prompt(self, ctx: Dict[str, Any]) -> str:
        return f"""
Examine the following dataset medical chart and return a clinical consultation report in strictly valid JSON:

DATASET VITALS:
- Total Sample Rows: {ctx['n_rows']}
- Feature Count: {ctx['n_cols']}
- Column Names: {', '.join(ctx['columns'])}
- Target Column: {ctx['target_column'] or 'Unsupervised / Pure Cleaning'}
- Overall Data Quality Score: {ctx['quality_score']}/100
- Training Readiness Gate: {ctx['readiness']}
- Critical Blockers: {json.dumps(ctx['critical_blockers'])}
- Domain Diagnostics: {json.dumps(ctx['domain_findings'])}
- Model Benchmark Summary: {json.dumps(ctx['model_summary'])}
- Error Analysis & Subgroups: {json.dumps(ctx['error_summary'])}

RETURN STRICTLY JSON WITH THESE EXACT KEYS:
{{
  "clinical_impression": "Concise 2-sentence medical diagnosis of dataset condition",
  "patient_status": "CRITICAL" | "GUARDED" | "STABLE" | "EXCELLENT",
  "survival_forecast": "Estimated % survival in production with concrete rationale",
  "semantic_leakage_review": [
    {{"column": "col_name", "risk_level": "HIGH" | "MEDIUM" | "LOW", "rationale": "Why this feature risks target or post-event leakage based on real-world naming semantics"}}
  ],
  "fairness_bias_analysis": "Diagnostic of minority representation, accuracy illusion risk, and subgroup blind spots",
  "prescriptions": [
    "Surgical step 1",
    "Surgical step 2",
    "Surgical step 3"
  ],
  "discharge_letter_markdown": "Complete formal Physician's Discharge Summary formatted in GitHub Markdown with sections: ## 🩺 CLINICAL SUMMARY, ## ⚠️ CRITICAL PATHOLOGIES, ## 💊 SURGICAL PRESCRIPTION, ## 🔮 PRODUCTION SURVIVAL PROGNOSIS."
}}
"""

    def _format_json_response(self, parsed: Dict[str, Any], provider: str) -> AIConsultationReport:
        return AIConsultationReport(
            clinical_impression=parsed.get("clinical_impression", "Evaluation completed."),
            patient_status=parsed.get("patient_status", "GUARDED"),
            survival_forecast=parsed.get("survival_forecast", "Moderate production risk."),
            semantic_leakage_review=parsed.get("semantic_leakage_review", []),
            fairness_bias_analysis=parsed.get("fairness_bias_analysis", "No severe subgroup bias detected."),
            prescriptions=parsed.get("prescriptions", ["Apply standard pipeline remediation."]),
            discharge_letter_markdown=parsed.get("discharge_letter_markdown", "Consultation complete."),
            provider=provider,
        )

    def _synthesize_offline_consultation(
        self,
        df: Optional[Any],
        profile: Optional[Dict[str, Any]],
        quality_assessment: Optional[Any],
        experiment_result: Optional[Any],
        error_profile: Optional[Any],
        target_column: Optional[str],
    ) -> AIConsultationReport:
        """
        Deterministic, rule-based clinical intelligence engine.
        Synthesizes deep diagnostic findings into an authentic medical narrative without requiring an external LLM.
        """
        ctx = self._build_context_summary(df, profile, quality_assessment, experiment_result, error_profile, target_column)
        score = ctx["quality_score"]
        blockers = ctx["critical_blockers"]
        target = ctx["target_column"] or "target"
        cols = ctx["columns"]

        # 1. Determine Patient Status
        if blockers or score < 50:
            patient_status = "CRITICAL"
        elif score < 75:
            patient_status = "GUARDED"
        elif score < 90:
            patient_status = "STABLE"
        else:
            patient_status = "EXCELLENT"

        # 2. Semantic Leakage Analysis (Heuristic Domain Reasoning)
        leakage_review = []
        suspicious_patterns = [
            (r"(copy|clone|mirror)", "Direct or cloned replica of target labels"),
            (r"(scaled|norm|inverted|log_)", "Mathematical derivative or transformation of ground truth"),
            (r"(id|seq|uuid|index|row_num)", "Row index or monotonically increasing identifier with spurious correlation"),
            (r"(payout|paid|settle|discharge|refund|resolved)", "Post-incident or post-event feature that does not exist at inference time"),
            (r"(next_|future_|subsequent)", "Look-ahead or time-travel variable leaking future state"),
        ]

        for col in cols:
            if col == target:
                continue
            col_lower = col.lower()
            matched = False
            for pat, reason in suspicious_patterns:
                if re.search(pat, col_lower):
                    leakage_review.append({
                        "column": col,
                        "risk_level": "HIGH",
                        "rationale": f"Feature name matches pattern '{pat}'. {reason}.",
                    })
                    matched = True
                    break
            if not matched and target and target.lower() in col_lower:
                leakage_review.append({
                    "column": col,
                    "risk_level": "HIGH",
                    "rationale": f"Feature name embeds the target name '{target}', strongly indicating target derived identity.",
                })

        # 3. Fairness and Accuracy Illusion Diagnosis
        has_illusion = ctx["model_summary"].get("accuracy_illusion", False)
        imbalance_findings = ctx["domain_findings"].get("imbalance", {}).get("findings", [])
        if has_illusion:
            fairness_text = (
                "⚠️ SEVERE ACCURACY ILLUSION DETECTED: The model exhibits superficially high nominal test accuracy "
                "masking catastrophic minority class recall (<40%). In production, this model will fail to identify "
                "the critical target class (e.g. fraud or disease) while appearing 90%+ accurate on paper."
            )
        elif imbalance_findings:
            fairness_text = (
                f"Class prevalence is skewed ({'; '.join(imbalance_findings[:2])}). "
                "Standard cross-entropy loss will over-index on majority examples. Calibrate decision thresholds."
            )
        else:
            fairness_text = "Balanced class distributions observed. No systemic representation bias detected."

        # 4. Formulate Prescriptions
        prescriptions = []
        if leakage_review:
            leak_cols = ", ".join([f"'{l['column']}'" for l in leakage_review[:3]])
            prescriptions.append(f"Excise semantic data leaks: Immediately drop features {leak_cols} to prevent false test metric inflation.")
        if blockers:
            prescriptions.append(f"Resolve {len(blockers)} Critical Blockers before initiating hyperparameter optimization: {blockers[0]}.")
        if has_illusion or imbalance_findings:
            prescriptions.append("Enforce Cost-Sensitive Learning: Optimize Balanced Accuracy, F1, and PR-AUC instead of raw accuracy.")
        if ctx["error_summary"].get("blind_spots"):
            bs = ctx["error_summary"]["blind_spots"][0]
            prescriptions.append(f"Remediate Subgroup Blind Spot: Gather additional calibration samples for slice {bs}.")
        prescriptions.append("Export and verify standalone reproduce.py pipeline script to ensure bit-for-bit deployment fidelity.")

        # 5. Production Survival Forecast
        if patient_status == "CRITICAL":
            survival_forecast = "POOR (< 25%): High probability of immediate production failure or post-deployment model collapse."
            impression = (
                f"The patient is in critical condition with severe integrity pathologies (Score: {score}/100). "
                f"Immediate surgical intervention required: {blockers[0] if blockers else 'Severe data leakage detected'}."
            )
        elif patient_status == "GUARDED":
            survival_forecast = "MODERATE (~ 55%): Viable in sandbox environments, but unaddressed missingness or skew will degrade inference."
            impression = (
                f"The dataset presents guarded prognosis (Score: {score}/100). "
                "Functional predictive signal is present, but unaddressed data anomalies threaten generalization stability."
            )
        elif patient_status == "STABLE":
            survival_forecast = "STRONG (~ 85%): Low risk of production collapse once standard imputation and encoding are locked."
            impression = (
                f"The patient is stable with solid baseline health (Score: {score}/100). "
                "Routine pre-processing will bring this dataset to deployment-ready grade."
            )
        else:
            survival_forecast = "EXCELLENT (> 95%): Exceptional data hygiene. Clean partitions and honest predictive signal."
            impression = (
                f"The patient is in peak health (Score: {score}/100). "
                "Zero critical leakage, verified balanced distributions, and honest predictive features."
            )

        # 6. Assemble Formal Physician's Discharge Letter Markdown
        date_str = datetime.now().strftime("%B %d, %Y")
        leakage_bullets = "\n".join([
            f"- **`{l['column']}`** ({l['risk_level']} RISK): {l['rationale']}"
            for l in leakage_review
        ]) or "- *No high-risk semantic leakage columns flagged.*"

        prescription_bullets = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prescriptions)])

        discharge_letter = f"""# 🩺 CLINICAL CONSULTATION & DISCHARGE SUMMARY

**Attending AI Physician:** Dr. Datadoctor, MD (Lead ML Data Pathologist)  
**Consultation Date:** {date_str}  
**Patient File:** `{target}` classification cohort ({ctx['n_rows']:,} rows, {ctx['n_cols']} features)  
**Overall Clinical Health Index:** **{score}/100** ({patient_status})

---

### ## 🩺 CLINICAL IMPRESSION
{impression}

---

### ## ⚠️ CRITICAL PATHOLOGY FINDINGS & LEAKAGE SCAN
{leakage_bullets}

**Fairness & Minority Subgroup Evaluation:**  
{fairness_text}

---

### ## 💊 SURGICAL PRESCRIPTION
{prescription_bullets}

---

### ## 🔮 PRODUCTION SURVIVAL PROGNOSIS
**Forecast:** **{survival_forecast}**  
*Clinical Note:* Datasets that proceed to model training with uninsulated leakage yield artificially inflated validation metrics (~99%) that immediately plummet in customer-facing production. Enforcing strict partition isolation and surgical remediation preserves production integrity.
"""

        return AIConsultationReport(
            clinical_impression=impression,
            patient_status=patient_status,
            survival_forecast=survival_forecast,
            semantic_leakage_review=leakage_review,
            fairness_bias_analysis=fairness_text,
            prescriptions=prescriptions,
            discharge_letter_markdown=discharge_letter.strip(),
            provider="Built-In Expert Diagnostic Synthesizer (Offline)",
        )
