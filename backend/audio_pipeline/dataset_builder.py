import os
import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_builder")


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


def build_manifests(
    transcripts_file: Optional[str] = None,
    manifests_dir: Optional[str] = None,
    config_path: str = "config.yaml",
    include_pending: bool = False,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Builds training, validation, and test manifests in JSONL & CSV format.
    Ensures 80/10/10 split and NeMo Parakeet / PyTorch compatibility.
    """
    config = load_config(config_path)
    t_file = transcripts_file or str(Path(config.get("dataset", {}).get("transcripts_dir", "datasets/transcripts")) / "transcripts_master.json")
    m_dir = manifests_dir or config.get("dataset", {}).get("manifests_dir", "datasets/manifests")
    
    splits_cfg = config.get("dataset", {}).get("splits", {"train": 0.8, "validation": 0.1, "test": 0.1})
    seed = config.get("dataset", {}).get("random_seed", random_seed)
    
    m_path = Path(m_dir)
    m_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(t_file):
        raise FileNotFoundError(f"Transcripts master file not found: {t_file}")

    with open(t_file, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    # Filter records based on review status
    valid_records = []
    for r in records:
        if r.get("processing_status") != "success":
            continue
        
        # Use human corrected transcript if available, otherwise cleaned transcript
        target_text = r.get("corrected_transcript") or r.get("cleaned_transcript") or r.get("raw_transcript", "")
        if not target_text.strip():
            continue

        if not include_pending and r.get("review_status") == "rejected":
            continue

        valid_records.append(r)

    if not valid_records:
        logger.warning("No valid approved transcripts found to build dataset.")
        return {"train_count": 0, "val_count": 0, "test_count": 0, "total": 0}

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = list(valid_records)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    train_ratio = splits_cfg.get("train", 0.8)
    val_ratio = splits_cfg.get("validation", 0.1)

    n_train = max(1, int(round(n_total * train_ratio))) if n_total >= 3 else n_total
    n_val = max(1, int(round(n_total * val_ratio))) if (n_total >= 3 and n_total - n_train > 1) else (1 if n_total > 1 else 0)
    
    if n_train + n_val > n_total:
        n_train = max(1, n_total - n_val)
        
    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]
    
    # If test_set is empty but total > 2, adjust
    if not test_set and len(val_set) > 1:
        test_set = [val_set.pop()]

    logger.info(f"Splits created: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)} (Total={n_total})")

    # Helper function to write NeMo JSONL format
    def format_nemo_record(r: Dict[str, Any]) -> Dict[str, Any]:
        text_content = r.get("corrected_transcript") or r.get("cleaned_transcript") or r.get("raw_transcript", "")
        item = {
            "audio_filepath": os.path.abspath(r["audio_filepath"]),
            "duration": float(r.get("duration", 0.0)),
            "text": text_content.strip(),
            "raw_text": r.get("raw_transcript", "").strip(),
            "language": r.get("language", "unknown"),
            "sample_rate": int(r.get("sample_rate", 16000)),
            "speaker_count": r.get("speaker_count"),
            "speaker_labels": r.get("speaker_labels"),
            "confidence": r.get("confidence"),
            "verified_by_human": r.get("verified_by_human", False),
            "review_status": r.get("review_status", "pending"),
        }
        return item

    def write_jsonl(file_path: Path, dataset_list: List[Dict[str, Any]]):
        with open(file_path, "w", encoding="utf-8") as f_out:
            for item in dataset_list:
                nemo_rec = format_nemo_record(item)
                f_out.write(json.dumps(nemo_rec, ensure_ascii=False) + "\n")

    # 1. Write full dataset JSONL
    full_jsonl_path = m_path / "dataset.jsonl"
    write_jsonl(full_jsonl_path, shuffled)

    # 2. Write split JSONLs
    train_jsonl_path = m_path / "train.jsonl"
    val_jsonl_path = m_path / "validation.jsonl"
    test_jsonl_path = m_path / "test.jsonl"

    write_jsonl(train_jsonl_path, train_set)
    write_jsonl(val_jsonl_path, val_set)
    write_jsonl(test_jsonl_path, test_set)

    # 3. Write CSV format for easy tabular inspection
    csv_path = m_path / "dataset.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        fieldnames = [
            "id", "audio_filepath", "duration", "sample_rate", 
            "raw_transcript", "cleaned_transcript", "corrected_transcript",
            "language", "confidence", "review_status", "verified_by_human", "split"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for s_name, s_data in [("train", train_set), ("validation", val_set), ("test", test_set)]:
            for r in s_data:
                writer.writerow({
                    "id": r.get("id", ""),
                    "audio_filepath": r.get("audio_filepath", ""),
                    "duration": r.get("duration", 0.0),
                    "sample_rate": r.get("sample_rate", 16000),
                    "raw_transcript": r.get("raw_transcript", ""),
                    "cleaned_transcript": r.get("cleaned_transcript", ""),
                    "corrected_transcript": r.get("corrected_transcript", ""),
                    "language": r.get("language", ""),
                    "confidence": r.get("confidence", ""),
                    "review_status": r.get("review_status", ""),
                    "verified_by_human": r.get("verified_by_human", False),
                    "split": s_name,
                })

    return {
        "total_records": n_total,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "manifests": {
            "dataset_jsonl": str(full_jsonl_path),
            "train_jsonl": str(train_jsonl_path),
            "val_jsonl": str(val_jsonl_path),
            "test_jsonl": str(test_jsonl_path),
            "dataset_csv": str(csv_path)
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset & Manifest Builder for STT")
    parser.add_argument("--transcripts", type=str, default="datasets/transcripts/transcripts_master.json")
    parser.add_argument("--manifests_dir", type=str, default="datasets/manifests")
    parser.add_argument("--include_pending", action="store_true", help="Include pending review records")
    args = parser.parse_args()

    res = build_manifests(
        transcripts_file=args.transcripts,
        manifests_dir=args.manifests_dir,
        include_pending=args.include_pending
    )
    print("Manifests created:", json.dumps(res, indent=2))
