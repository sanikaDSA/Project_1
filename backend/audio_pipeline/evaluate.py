import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio_pipeline.normalizer import validate_medical_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_stt")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        base_dir = Path(__file__).resolve().parent.parent
        alt_path = base_dir / "config.yaml"
        if alt_path.exists():
            config_path = str(alt_path)
        else:
            return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance between reference and hypothesis tokens.
    Returns (substitutions, deletions, insertions, total_ref_tokens).
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrace to find S, D, I counts
    i, j = n, m
    substitutions, deletions, insertions = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            break

    return substitutions, deletions, insertions, n


def calculate_wer_cer(
    references: List[str], 
    hypotheses: List[str]
) -> Dict[str, Any]:
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER).
    Supports jiwer if installed, with native pure-python fallback.
    """
    try:
        import jiwer
        # Try jiwer library
        wer_val = float(jiwer.wer(references, hypotheses))
        cer_val = float(jiwer.cer(references, hypotheses))
    except Exception:
        # Pure Python fallback
        total_words_ref = 0
        total_word_errors = 0
        total_chars_ref = 0
        total_char_errors = 0

        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.strip().split()
            hyp_words = hyp.strip().split()
            s_w, d_w, i_w, n_w = compute_edit_distance(ref_words, hyp_words)
            total_word_errors += (s_w + d_w + i_w)
            total_words_ref += max(1, n_w)

            ref_chars = list(ref.strip())
            hyp_chars = list(hyp.strip())
            s_c, d_c, i_c, n_c = compute_edit_distance(ref_chars, hyp_chars)
            total_char_errors += (s_c + d_c + i_c)
            total_chars_ref += max(1, n_c)

        wer_val = round(total_word_errors / max(1, total_words_ref), 4)
        cer_val = round(total_char_errors / max(1, total_chars_ref), 4)

    return {
        "wer": round(wer_val, 4),
        "cer": round(cer_val, 4),
        "accuracy_word": round(max(0.0, 1.0 - wer_val) * 100, 2),
        "accuracy_char": round(max(0.0, 1.0 - cer_val) * 100, 2),
    }


def evaluate_predictions(
    test_records: List[Dict[str, Any]],
    reports_dir: Optional[str] = None,
    config_path: str = "config.yaml"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation report generator.
    Evaluates:
    - WER / CER
    - Medical Term Precision & Recall
    - Sentence-by-sentence comparison
    """
    config = load_config(config_path)
    r_dir = reports_dir or config.get("paths", {}).get("reports_dir", "outputs/reports")
    out_path = Path(r_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    references = []
    hypotheses = []
    sample_evaluations = []

    med_ref_total = 0
    med_hyp_matched = 0

    for idx, item in enumerate(test_records):
        ref_text = item.get("reference") or item.get("corrected_transcript") or item.get("text", "")
        hyp_text = item.get("hypothesis") or item.get("predicted_transcript", "")
        audio_fp = item.get("audio_filepath", f"sample_{idx}")

        if not ref_text.strip():
            continue

        references.append(ref_text)
        hypotheses.append(hyp_text)

        # Single item metrics
        item_metrics = calculate_wer_cer([ref_text], [hyp_text])
        
        # Medical terms extraction & check
        ref_meds = validate_medical_terms(ref_text)
        hyp_meds = validate_medical_terms(hyp_text)
        
        med_ref_total += len(ref_meds)
        med_matched = set(ref_meds).intersection(set(hyp_meds))
        med_hyp_matched += len(med_matched)

        sample_evaluations.append({
            "sample_id": idx + 1,
            "audio_filepath": audio_fp,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "wer": item_metrics["wer"],
            "cer": item_metrics["cer"],
            "reference_medical_terms": ref_meds,
            "detected_medical_terms": hyp_meds,
            "retained_medical_terms": list(med_matched),
        })

    if not references:
        logger.warning("No reference-hypothesis pairs to evaluate.")
        return {"error": "No valid pairs found"}

    overall_metrics = calculate_wer_cer(references, hypotheses)
    
    med_recall = round((med_hyp_matched / max(1, med_ref_total)) * 100, 2)

    report = {
        "total_samples": len(references),
        "overall_wer": overall_metrics["wer"],
        "overall_cer": overall_metrics["cer"],
        "word_accuracy_pct": overall_metrics["accuracy_word"],
        "char_accuracy_pct": overall_metrics["accuracy_char"],
        "medical_terms_total_in_reference": med_ref_total,
        "medical_terms_accurately_retained": med_hyp_matched,
        "medical_terminology_retention_rate_pct": med_recall,
        "sample_evaluations": sample_evaluations,
    }

    # Save JSON Report
    json_path = out_path / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Save Human-readable Summary TXT
    summary_path = out_path / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=================================================================\n")
        f.write("       DOCTOR-PATIENT SPEECH-TO-TEXT EVALUATION REPORT          \n")
        f.write("=================================================================\n\n")
        f.write(f"Total Test Samples Evaluated:      {len(references)}\n")
        f.write(f"Word Error Rate (WER):             {overall_metrics['wer']:.4f} ({overall_metrics['accuracy_word']}% accuracy)\n")
        f.write(f"Character Error Rate (CER):        {overall_metrics['cer']:.4f} ({overall_metrics['accuracy_char']}% accuracy)\n")
        f.write(f"Medical Terminology Accuracy:      {med_recall}% ({med_hyp_matched}/{med_ref_total} terms preserved)\n\n")
        f.write("-----------------------------------------------------------------\n")
        f.write("SAMPLE CLINICAL PREDICTIONS:\n")
        f.write("-----------------------------------------------------------------\n")
        for s in sample_evaluations[:10]:
            f.write(f"Sample #{s['sample_id']}:\n")
            f.write(f"  Audio: {s['audio_filepath']}\n")
            f.write(f"  REF:   {s['reference']}\n")
            f.write(f"  HYP:   {s['hypothesis']}\n")
            f.write(f"  WER:   {s['wer']:.2f} | CER: {s['cer']:.2f}\n")
            f.write(f"  Meds:  {s['retained_medical_terms']} / {s['reference_medical_terms']}\n\n")

    logger.info(f"Evaluation report generated at {json_path} and {summary_path}")
    return report


if __name__ == "__main__":
    test_data = [
        {
            "audio_filepath": "datasets/processed_audio/sample1_16k_mono.wav",
            "reference": "डॉक्टर मला शुगर आहे. Fasting blood sugar 180 आणि Metformin 500 mg चालू आहे.",
            "hypothesis": "डॉक्टर मला शुगर आहे. Fasting blood sugar 180 आणि Metformin 500 mg चालू आहे."
        },
        {
            "audio_filepath": "datasets/processed_audio/sample2_16k_mono.wav",
            "reference": "Patient complains of chest pain and high BP 150/90 mmHg. Prescribed Telmisartan 40 mg OD.",
            "hypothesis": "Patient complains of chest pain and high BP 150/90. Prescribed Telmisartan 40 mg OD."
        }
    ]
    res = evaluate_predictions(test_data)
    print("WER:", res["overall_wer"], "| CER:", res["overall_cer"])
