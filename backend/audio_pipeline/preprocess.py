import os
import sys
import glob
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audio_preprocess")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        # Search relative to script location
        base_dir = Path(__file__).resolve().parent.parent
        alt_path = base_dir / "config.yaml"
        if alt_path.exists():
            config_path = str(alt_path)
        else:
            return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_audio_files(
    directory: str, 
    supported_extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Recursively find all supported audio files in the given directory.
    Supported: .wav, .mp3, .m4a, .flac
    """
    if supported_extensions is None:
        supported_extensions = [".wav", ".mp3", ".m4a", ".flac"]
    
    supported_ext_set = {ext.lower() for ext in supported_extensions}
    audio_files: List[Path] = []
    
    root_path = Path(directory)
    if not root_path.exists():
        logger.warning(f"Audio directory does not exist: {directory}")
        return []

    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_ext_set:
            audio_files.append(file_path)
            
    logger.info(f"Found {len(audio_files)} audio file(s) in {directory}")
    return sorted(audio_files)


def read_audio_file(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Read audio file using miniaudio, soundfile, torchaudio, or wave/scipy fallback.
    Returns (audio_data: float32 np.ndarray in [-1, 1], sample_rate: int).
    """
    # 1. Try miniaudio (universal MP3, WAV, FLAC decoder)
    try:
        import miniaudio
        decoded = miniaudio.decode_file(str(file_path), dither=miniaudio.DitherMode.NONE)
        data = np.array(decoded.samples, dtype=np.float32)
        if decoded.nchannels > 1:
            data = data.reshape(-1, decoded.nchannels)
        return data, decoded.sample_rate
    except Exception as e0:
        pass

    # 2. Try soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(str(file_path), dtype="float32", always_2d=False)
        return data, sr
    except Exception as e1:
        pass

    # 2. Try torchaudio if installed
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(file_path))
        data = waveform.numpy()
        if data.ndim == 2:
            # shape (channels, samples) -> transpose to (samples, channels) or (samples,)
            data = data.T if data.shape[0] > 1 else data[0]
        return data, sr
    except Exception as e2:
        pass

    # 3. Try scipy.io.wavfile for WAV files
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(file_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
        return data, sr
    except Exception as e3:
        pass

    raise RuntimeError(
        f"Failed to read audio file '{file_path}'. Formats supported: WAV, MP3, M4A, FLAC."
    )


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D audio array using scipy or numpy interpolation."""
    if orig_sr == target_sr:
        return data
    from scipy import signal
    num_target_samples = int(round(len(data) * float(target_sr) / orig_sr))
    resampled = signal.resample(data, num_target_samples)
    return resampled.astype(np.float32)


def convert_to_mono(data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if data.ndim == 1:
        return data
    elif data.ndim == 2:
        # (samples, channels)
        return np.mean(data, axis=1, dtype=np.float32)
    else:
        flat = data.reshape(-1, data.shape[-1])
        return np.mean(flat, axis=1, dtype=np.float32)


def normalize_audio(data: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Safely normalize audio to target peak value without clipping.
    """
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        normalized = (data / max_val) * target_peak
    else:
        normalized = data
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def write_wav(file_path: Path, data: np.ndarray, sample_rate: int) -> None:
    """Write float32 audio as 16-bit PCM WAV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(file_path), data, sample_rate, subtype="PCM_16")
    except Exception:
        from scipy.io import wavfile
        int16_data = (data * 32767.0).astype(np.int16)
        wavfile.write(str(file_path), sample_rate, int16_data)


def preprocess_single_audio(
    input_path: Path,
    output_dir: Path,
    target_sr: int = 16000,
    target_peak: float = 0.95,
) -> Dict[str, Any]:
    """
    Preprocess a single audio file:
    1. Read original audio
    2. Convert to mono
    3. Resample to 16 kHz
    4. Normalize safely
    5. Save as standardized 16kHz mono WAV in output_dir
    6. Return metadata dictionary
    """
    output_filename = f"{input_path.stem}_16k_mono.wav"
    output_path = output_dir / output_filename

    # Read audio
    data, orig_sr = read_audio_file(input_path)
    
    # Mono conversion
    mono_data = convert_to_mono(data)
    
    # Resample to 16kHz
    resampled_data = resample_audio(mono_data, orig_sr, target_sr)
    
    # Normalize
    normalized_data = normalize_audio(resampled_data, target_peak)
    
    # Save output
    write_wav(output_path, normalized_data, target_sr)
    
    duration = float(len(normalized_data)) / float(target_sr)
    
    return {
        "original_file": str(input_path.resolve()),
        "processed_file": str(output_path.resolve()),
        "filename": output_filename,
        "original_sr": int(orig_sr),
        "sample_rate": int(target_sr),
        "channels": 1,
        "duration": round(duration, 3),
        "status": "success",
    }


def preprocess_dataset(
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    error_log_path: Optional[str] = None,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """
    Process entire raw dataset folder recursively.
    Skips corrupted files, logs errors, preserves original files.
    """
    config = load_config(config_path)
    
    raw_dir_str = raw_dir or config.get("dataset", {}).get("raw_audio_dir", "datasets/raw_audio")
    output_dir_str = output_dir or config.get("dataset", {}).get("processed_audio_dir", "datasets/processed_audio")
    error_log_str = error_log_path or config.get("paths", {}).get("error_log", "outputs/logs/pipeline_errors.log")
    
    target_sr = config.get("audio", {}).get("target_sample_rate", 16000)
    target_peak = config.get("audio", {}).get("normalization_peak", 0.95)
    supported_exts = config.get("audio", {}).get("supported_extensions", [".wav", ".mp3", ".m4a", ".flac"])

    raw_path = Path(raw_dir_str)
    out_path = Path(output_dir_str)
    out_path.mkdir(parents=True, exist_ok=True)
    
    err_log_file = Path(error_log_str)
    err_log_file.parent.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(str(raw_path), supported_exts)
    
    processed_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, Any]] = []

    for file_path in audio_files:
        try:
            record = preprocess_single_audio(
                input_path=file_path,
                output_dir=out_path,
                target_sr=target_sr,
                target_peak=target_peak,
            )
            processed_records.append(record)
            logger.info(f"Successfully processed: {file_path.name} -> {record['filename']} ({record['duration']}s)")
        except Exception as exc:
            err_msg = f"Failed to preprocess '{file_path}': {str(exc)}\n{traceback.format_exc()}"
            logger.error(err_msg)
            with open(err_log_file, "a", encoding="utf-8") as f:
                f.write(f"--- ERROR: {file_path} ---\n{err_msg}\n\n")
            error_records.append({
                "file": str(file_path),
                "error": str(exc),
                "status": "failed",
            })

    summary = {
        "total_detected": len(audio_files),
        "total_processed": len(processed_records),
        "total_failed": len(error_records),
        "processed_records": processed_records,
        "error_records": error_records,
    }
    
    logger.info(f"Preprocessing completed: {len(processed_records)} success, {len(error_records)} failed.")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio Preprocessing Pipeline for Doctor-Patient STT")
    parser.add_argument("--raw_dir", type=str, default="datasets/raw_audio", help="Path to raw audio folder")
    parser.add_argument("--output_dir", type=str, default="datasets/processed_audio", help="Path to save processed WAVs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    res = preprocess_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir, config_path=args.config)
    print(f"Done. Processed: {res['total_processed']}/{res['total_detected']}")
