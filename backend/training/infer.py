import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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

from training.dataset import Vocabulary
from training.train import DoctorPatientASRModel, load_config
from audio_pipeline.preprocess import read_audio_file, convert_to_mono, resample_audio, normalize_audio
from audio_pipeline.normalizer import clean_medical_transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stt_infer")


class OfflineSTTInference:
    """
    Offline STT Inference engine using trained checkpoint.
    """
    def __init__(self, checkpoint_path: str = "outputs/checkpoints/best_model.pt", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path} onto {self.device}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load vocab
        self.vocab = Vocabulary()
        if "vocab" in checkpoint:
            self.vocab.char2idx = checkpoint["vocab"]
            self.vocab.idx2char = {v: k for k, v in checkpoint["vocab"].items()}

        self.model = DoctorPatientASRModel(
            n_mels=80,
            encoder_dim=256,
            num_layers=4,
            vocab_size=len(self.vocab)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def compute_features(self, audio: np.ndarray, sample_rate: int = 16000) -> Any:
        try:
            import torchaudio.transforms as T
            wav_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=80
            )
            mel = mel_transform(wav_tensor)
            log_mel = torch.log(mel + 1e-6).squeeze(0).transpose(0, 1)
            return log_mel
        except Exception:
            from scipy import signal
            f, t, Sxx = signal.spectrogram(audio, fs=sample_rate, nperseg=400, noverlap=240, nfft=512)
            log_spec = np.log(Sxx[:80, :] + 1e-6).T
            return torch.from_numpy(log_spec).float()

    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        path = Path(audio_file)
        data, orig_sr = read_audio_file(path)
        data = convert_to_mono(data)
        if orig_sr != 16000:
            data = resample_audio(data, orig_sr, 16000)
        data = normalize_audio(data, 0.95)

        features = self.compute_features(data, 16000).unsqueeze(0).to(self.device) # (1, T, 80)
        lengths = torch.tensor([features.size(1)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            log_probs, out_lengths, logits = self.model(features, lengths)
            preds = torch.argmax(logits, dim=-1)
            pred_ids = preds[0, :out_lengths[0].item()].cpu().tolist()
            raw_text = self.vocab.decode_ctc(pred_ids)

        norm_res = clean_medical_transcript(raw_text)

        return {
            "audio_file": str(path.resolve()),
            "raw_transcription": norm_res["raw_transcript"],
            "cleaned_transcription": norm_res["cleaned_transcript"],
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Offline STT Inference on Audio File")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt", help="Path to best_model.pt")
    args = parser.parse_args()

    engine = OfflineSTTInference(checkpoint_path=args.checkpoint)
    res = engine.transcribe(args.audio)
    print("Inference Result:")
    print("Raw:    ", res["raw_transcription"])
    print("Cleaned:", res["cleaned_transcription"])
