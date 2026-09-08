import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

# Initialize DLL directories for PyTorch on Windows
lib_dir = Path(sys.prefix) / "site-packages" / "torch" / "lib"
user_lib_dir = Path(os.getenv("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages" / "torch" / "lib"
for p in [str(lib_dir), str(user_lib_dir), sys.prefix, os.path.dirname(sys.executable)]:
    if os.path.exists(p) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import Vocabulary, ASRDataset, asr_collate_fn
from audio_pipeline.evaluate import calculate_wer_cer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stt_trainer")


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


_ModuleBase = nn.Module if nn is not None else object

class ConvSubsampling(_ModuleBase):
    """2D Convolutional Subsampling layer to compress time by 4x."""
    def __init__(self, in_channels: int = 80, out_channels: int = 256):
        if nn is None:
            return
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out_dim = 32 * (in_channels // 4)
        self.proj = nn.Linear(self.out_dim, out_channels)

    def forward(self, x: Any, lengths: Any):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * d)
        out = self.proj(x)
        out_lengths = ((lengths + 1) // 2 + 1) // 2
        return out, out_lengths


class DoctorPatientASRModel(_ModuleBase):
    """
    Offline Trainable Acoustic Model for Multilingual Doctor-Patient Conversations.
    Architecture: ConvSubsampling + BiLSTM/Conformer Encoder + CTC Projection Head.
    """
    def __init__(self, n_mels: int = 80, encoder_dim: int = 256, num_layers: int = 4, vocab_size: int = 200):
        if nn is None:
            return
        super().__init__()
        self.subsampling = ConvSubsampling(in_channels=n_mels, out_channels=encoder_dim)
        
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=encoder_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.classifier = nn.Linear(encoder_dim, vocab_size)

    def forward(self, features: Any, lengths: Any):
        x, out_lengths = self.subsampling(features, lengths)
        lstm_out, _ = self.lstm(x)
        norm_out = self.layer_norm(lstm_out)
        logits = self.classifier(norm_out)
        log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
        return log_probs, out_lengths, logits


def train_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    scaler: Any,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                log_probs, out_lengths, _ = model(features, feature_lengths)
                loss = criterion(log_probs, targets, out_lengths, target_lengths)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            log_probs, out_lengths, _ = model(features, feature_lengths)
            loss = criterion(log_probs, targets, out_lengths, target_lengths)
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / max(1, len(dataloader))


def validate_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    vocab: Any,
    device: Any,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            texts = batch["texts"]

            log_probs, out_lengths, logits = model(features, feature_lengths)
            loss = criterion(log_probs, targets, out_lengths, target_lengths)
            total_loss += loss.item()

            # Greedy decoding
            preds = torch.argmax(logits, dim=-1) # (B, T_sub)
            for i in range(preds.size(0)):
                seq_len = out_lengths[i].item()
                pred_ids = preds[i, :seq_len].cpu().tolist()
                decoded_text = vocab.decode_ctc(pred_ids)
                all_hyps.append(decoded_text)
                all_refs.append(texts[i])

    avg_loss = total_loss / max(1, len(dataloader))
    metrics = calculate_wer_cer(all_refs, all_hyps)

    return {
        "val_loss": avg_loss,
        "val_wer": metrics["wer"],
        "val_cer": metrics["cer"],
        "references": all_refs,
        "hypotheses": all_hyps,
    }


def run_training_pipeline(
    config_path: str = "config.yaml",
    resume_path: Optional[str] = None,
    num_epochs_override: Optional[int] = None
) -> Dict[str, Any]:
    config = load_config(config_path)
    t_cfg = config.get("training", {})
    p_cfg = config.get("paths", {})
    m_cfg = config.get("dataset", {})

    # GPU / Device Auto Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    use_amp = bool(t_cfg.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) enabled.")

    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path(p_cfg.get("logs_dir", "outputs/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "training_metrics.jsonl"

    # Initialize Vocabulary & Datasets
    vocab = Vocabulary()
    manifests_dir = m_cfg.get("manifests_dir", "datasets/manifests")
    train_manifest = str(Path(manifests_dir) / "train.jsonl")
    val_manifest = str(Path(manifests_dir) / "validation.jsonl")

    train_dataset = ASRDataset(train_manifest, vocab)
    val_dataset = ASRDataset(val_manifest, vocab)

    if len(train_dataset) == 0:
        logger.warning("Train dataset manifest is empty. Please run preprocess, transcribe, and build-dataset first.")
        return {"status": "error", "message": "Train manifest empty"}

    batch_size = min(len(train_dataset), t_cfg.get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=asr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=max(1, min(len(val_dataset), batch_size)), shuffle=False, collate_fn=asr_collate_fn) if len(val_dataset) > 0 else None

    # Model & Optimizer
    model = DoctorPatientASRModel(
        n_mels=80,
        encoder_dim=t_cfg.get("encoder_dim", 256),
        num_layers=t_cfg.get("num_encoder_layers", 4),
        vocab_size=len(vocab)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(t_cfg.get("learning_rate", 3e-4)), 
        weight_decay=float(t_cfg.get("weight_decay", 1e-4))
    )
    criterion = nn.CTCLoss(blank=vocab.char2idx["<blank>"], zero_infinity=True)

    start_epoch = 1
    best_wer = float("inf")

    # Check for resume
    resume_target = resume_path or t_cfg.get("resume_checkpoint")
    if resume_target and os.path.exists(resume_target):
        logger.info(f"Resuming training from checkpoint: {resume_target}")
        checkpoint = torch.load(resume_target, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_wer = checkpoint.get("best_wer", float("inf"))

    total_epochs = num_epochs_override or t_cfg.get("num_epochs", 15)
    grad_accum_steps = t_cfg.get("gradient_accumulation_steps", 1)

    logger.info(f"Starting training: {total_epochs} epochs, batch size={batch_size}, vocab size={len(vocab)}")

    training_history = []

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, grad_accum_steps)
        
        if val_loader is not None and len(val_dataset) > 0:
            val_res = validate_epoch(model, val_loader, criterion, vocab, device)
            val_loss = val_res["val_loss"]
            val_wer = val_res["val_wer"]
            val_cer = val_res["val_cer"]
        else:
            val_loss = 0.0
            val_wer = 0.0
            val_cer = 0.0

        epoch_duration = round(time.time() - epoch_start, 2)
        
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_wer": round(val_wer, 4),
            "val_cer": round(val_cer, 4),
            "duration_sec": epoch_duration,
            "device": str(device),
        }
        training_history.append(epoch_metrics)

        logger.info(
            f"Epoch {epoch}/{total_epochs} [{epoch_duration}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val WER: {val_wer:.4f} | Val CER: {val_cer:.4f}"
        )

        # Write to log file
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(json.dumps(epoch_metrics) + "\n")

        # Save Regular Checkpoint
        ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "best_wer": best_wer,
            "vocab": vocab.char2idx,
        }, ckpt_path)

        # Save Best Model Checkpoint
        if val_wer < best_wer or epoch == 1:
            best_wer = val_wer
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": vocab.char2idx,
                "best_wer": best_wer,
                "val_wer": val_wer,
                "val_cer": val_cer,
            }, best_model_path)
            logger.info(f"--> Saved new BEST model checkpoint to {best_model_path} (WER: {val_wer:.4f})")

    return {
        "status": "completed",
        "total_epochs": total_epochs,
        "best_wer": best_wer,
        "best_model_path": str(checkpoint_dir / "best_model.pt"),
        "history": training_history,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline Doctor-Patient STT Model Trainer")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    run_training_pipeline(config_path=args.config, resume_path=args.resume, num_epochs_override=args.epochs)
