import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

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
from torch.utils.data import Dataset

from audio_pipeline.preprocess import read_audio_file, resample_audio, convert_to_mono, normalize_audio


# Character Vocabulary for Multilingual Indic (Devanagari) + English + Numerals + Punctuation
SPECIAL_TOKENS = ["<blank>", "<unk>", "<space>"]
DEFAULT_CHARS = (
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,-/:;!?'\""
    # Devanagari range (Marathi & Hindi)
    "ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्"
    "०१२३४५६७८९"
)

class Vocabulary:
    """Character-level vocabulary for CTC ASR."""
    def __init__(self, char_set: Optional[str] = None):
        chars = list(char_set or DEFAULT_CHARS)
        # Deduplicate while preserving order
        seen = set()
        unique_chars = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        self.char2idx: Dict[str, int] = {
            "<blank>": 0,
            "<unk>": 1,
            "<space>": 2,
        }
        self.idx2char: Dict[int, str] = {
            0: "<blank>",
            1: "<unk>",
            2: " ",
        }

        current_idx = 3
        for c in unique_chars:
            if c == " ":
                continue
            if c not in self.char2idx:
                self.char2idx[c] = current_idx
                self.idx2char[current_idx] = c
                current_idx += 1

    def encode(self, text: str) -> List[int]:
        ids = []
        for c in text:
            if c == " ":
                ids.append(self.char2idx["<space>"])
            elif c in self.char2idx:
                ids.append(self.char2idx[c])
            else:
                ids.append(self.char2idx["<unk>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        res = []
        for i in ids:
            if i == self.char2idx["<blank>"]:
                continue
            elif i == self.char2idx["<space>"]:
                res.append(" ")
            elif i in self.idx2char:
                res.append(self.idx2char[i])
        return "".join(res)

    def decode_ctc(self, ids: List[int]) -> str:
        """Greedy CTC collapse: remove consecutive duplicates and blanks."""
        collapsed = []
        prev = None
        for idx in ids:
            if idx != prev:
                if idx != self.char2idx["<blank>"]:
                    collapsed.append(idx)
            prev = idx
        return self.decode(collapsed)

    def __len__(self) -> int:
        return len(self.char2idx)


class ASRDataset(Dataset):
    """
    PyTorch Dataset loading NeMo-style JSONL manifests.
    Extracts 80-dim log Mel-Filterbank spectrogram features.
    """
    def __init__(
        self, 
        manifest_path: str, 
        vocab: Vocabulary, 
        sample_rate: int = 16000,
        n_mels: int = 80,
        max_duration: float = 30.0
    ):
        self.manifest_path = manifest_path
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.samples: List[Dict[str, Any]] = []

        if not os.path.exists(manifest_path):
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "audio_filepath" in item and ("text" in item or "raw_text" in item):
                        self.samples.append(item)
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def compute_features(self, audio: np.ndarray) -> Any:
        """Compute Log-Mel Spectrogram features (Time, Features)."""
        try:
            import importlib
            T = importlib.import_module("torchaudio.transforms")
            wav_tensor = torch.from_numpy(audio).float().unsqueeze(0) # (1, T)
            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=self.n_mels
            )
            mel = mel_transform(wav_tensor) # (1, n_mels, frames)
            log_mel = torch.log(mel + 1e-6).squeeze(0).transpose(0, 1) # (frames, n_mels)
            return log_mel
        except Exception:
            # Numpy / Scipy FFT fallback
            from scipy import signal
            f, t, Sxx = signal.spectrogram(audio, fs=self.sample_rate, nperseg=400, noverlap=240, nfft=512)
            log_spec = np.log(Sxx[:self.n_mels, :] + 1e-6).T # (frames, n_mels)
            return torch.from_numpy(log_spec).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        audio_path = item["audio_filepath"]
        text = item.get("text") or item.get("raw_text", "")

        # Read and preprocess audio
        try:
            data, orig_sr = read_audio_file(Path(audio_path))
            data = convert_to_mono(data)
            if orig_sr != self.sample_rate:
                data = resample_audio(data, orig_sr, self.sample_rate)
            data = normalize_audio(data, 0.95)
        except Exception as e:
            # Corrupted fallback to silence
            data = np.zeros(self.sample_rate, dtype=np.float32)

        features = self.compute_features(data)
        token_ids = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        return {
            "features": features,                  # (frames, n_mels)
            "targets": token_ids,                  # (target_len,)
            "feature_length": features.size(0),
            "target_length": token_ids.size(0),
            "text": text,
            "audio_filepath": audio_path,
        }


def asr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function with dynamic padding for variable audio and text lengths."""
    feature_lengths = [b["feature_length"] for b in batch]
    target_lengths = [b["target_length"] for b in batch]

    max_feat_len = max(feature_lengths)
    n_mels = batch[0]["features"].size(1)

    batch_size = len(batch)
    padded_features = torch.zeros(batch_size, max_feat_len, n_mels, dtype=torch.float32)

    all_targets = []
    texts = []
    filepaths = []

    for i, item in enumerate(batch):
        f_len = item["feature_length"]
        padded_features[i, :f_len, :] = item["features"]
        all_targets.append(item["targets"])
        texts.append(item["text"])
        filepaths.append(item["audio_filepath"])

    # Concatenate targets for CTC Loss or pad
    max_target_len = max(max(target_lengths), 1)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    for i, t in enumerate(all_targets):
        if len(t) > 0:
            padded_targets[i, :len(t)] = t

    return {
        "features": padded_features,                            # (B, T, n_mels)
        "feature_lengths": torch.tensor(feature_lengths, dtype=torch.long),  # (B,)
        "targets": padded_targets,                              # (B, L)
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),    # (B,)
        "texts": texts,
        "audio_filepaths": filepaths,
    }
